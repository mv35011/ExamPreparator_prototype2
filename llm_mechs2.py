# I created this because I was facing trouble to connect with Hugging face servers
#offline initialization of model

import os
import torch
import chromadb
import logging
import pytesseract
from PIL import Image
import cv2
import numpy as np
import fitz
import re
import PyPDF2

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import load_dataset, Dataset
from langchain_core.documents import Document

from file_adapter import StreamlitFileAdapter

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
import os
import json
from datasets import Dataset

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

class AIExamHelper:
    def __init__(self, model_path="./fine_tuned_mistral", db_path="./chroma_db"):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.embedding_model_path = "./models/all-MiniLM-L6-v2"
        self.base_model_path = "./models/Mistral-7B-Instruct-v0.1"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fs = StreamlitFileAdapter()
        self.tesseract_config = r'--psm 6 --oem 3'
        self.db_path = db_path
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.1"

        os.makedirs(db_path, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:

            if self.device == "cuda":
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                if free_memory < 3.5 * 1024 * 1024 * 1024:
                    logger.warning("Low GPU memory, falling back to CPU")
                    self.device = "cpu"

            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

            if os.path.exists(self.model_path):
                logger.info(f"Loading fine-model from {model_path}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,

                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                self.model = PeftModel.from_pretrained(base_model, model_path)
            else:
                logger.warning(f"Fine-tuned model not found at {model_path}, using base model")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,

                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if self.device == "cuda" else -1,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )

            self.generator_ready = True
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self.generator_ready = False

    def process_document(self, uploaded_file):
        try:
            filename = self.fs.save(uploaded_file.name, uploaded_file)
            file_path = self.fs.path(filename)

            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.pdf':
                extracted_text = self._extract_text_from_pdf(file_path)
            elif file_extension in ['.jpg', '.jpeg', '.png']:
                extracted_text = self._extract_text_from_image(file_path)
            else:
                self.fs.delete(filename)
                return {
                    'success': False,
                    'error': f"Unsupported file format: {file_extension}",
                    'extracted_text': ""
                }
            self.fs.delete(filename)

            if not extracted_text:
                return {
                    'success': False,
                    'error': "Could not extract text from the file"
                }
            processed_text = self._post_process_math_text(extracted_text)
            self._store_in_vectordb(processed_text, uploaded_file.name)

            return {
                'success': True,
                'filename': uploaded_file.name,
                'file_type': file_extension,
                'extracted_text': processed_text,
                'char_count': len(processed_text)
            }
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return {
                'success': False,
                'error': f"Error processing file: {str(e)}",
                'extracted_text': ""
            }

    def _extract_text_from_pdf(self, pdf_path):
        text = ""
        doc = None

        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text() + "\n"


            if not text.strip():
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text() + "\n"

            return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return self._extract_text_from_pdf_as_images(pdf_path)
        finally:

            if doc:
                doc.close()

    def _extract_text_from_pdf_as_images(self, pdf_path):
        doc = None
        try:
            text = ""
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(
                    img,
                    config=self.tesseract_config,
                    lang='eng+equ'
                )
                text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF-to-image extraction error: {str(e)}")
            return ""
        finally:

            if doc:
                doc.close()

    def _extract_text_from_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                image = Image.open(image_path)
                return pytesseract.image_to_string(
                    image,
                    config=self.tesseract_config,
                    lang="eng+equ"
                )

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            pil_img = Image.fromarray(thresh)

            text1 = pytesseract.image_to_string(
                pil_img,
                config=self.tesseract_config + ' -c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=()[]{}<>^\\∫∂∑∏√∞∈∋∩∪⊂⊃≤≥≠~πθα"',
                lang='eng+equ'
            )

            text2 = pytesseract.image_to_string(pil_img)

            return text1 if len(text1) > len(text2) else text2

        except Exception as e:
            logger.error(f"Image OCR error: {str(e)}")
            return ""

    def _post_process_math_text(self, text):

        replacements = {
            "integral": "∫",
            r"\int": "∫",
            r"\partial": "∂",
            "partial": "∂",
            r"\sum": "∑",
            "sum": "∑",
            r"\prod": "∏",
            r"\sqrt": "√",
            "sqrt": "√",
            "infinity": "∞",
            r"\infty": "∞",
            r"\pi": "π",
            "pi": "π",
            "theta": "θ",
            r"\theta": "θ",
            "alpha": "α",
            r"\alpha": "α",
        }

        processed_text = text


        pattern_cache = {}

        for wrong, correct in replacements.items():
            if wrong not in pattern_cache:
                if wrong.startswith('\\'):
                    pattern_cache[wrong] = re.compile(r'(?<![a-zA-Z])' + re.escape(wrong) + r'(?![a-zA-Z])')
                else:
                    pattern_cache[wrong] = re.compile(r'\b' + re.escape(wrong) + r'\b')

            processed_text = pattern_cache[wrong].sub(correct, processed_text)

        # Additional specific replacement for integral
        integral_pattern = re.compile(r'\\int(egrall?)?')
        processed_text = integral_pattern.sub('∫', processed_text)

        return processed_text

    def _store_in_vectordb(self, text, document_name):
        try:

            chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]

            documents = [
                Document(
                    page_content=chunk,
                    metadata={"source": document_name, "chunk": i}
                )
                for i, chunk in enumerate(chunks)
            ]
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} chunks from {document_name} to vector store")
            return True
        except Exception as e:
            logger.error(f"Error storing in vector database: {str(e)}")
            return False

    def fine_tune_model(self):
        try:

            with open("fine_tuning_data.json", "r") as f:
                training_data = json.load(f)

            formatted_data = [{
                "input": f"Context: {item['context']}\nGenerate a question and its step-by-step derivation.",
                "output": f"Question: {item['question']}\nDerivation: {item['derivation']}"
            } for item in training_data]

            training_dataset = Dataset.from_list(formatted_data)


            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                if free_memory < 3.5 * 1024 * 1024 * 1024:
                    logger.warning("Low GPU memory, falling back to CPU")
                    device_map = None
                    dtype = torch.float32
                else:
                    device_map = "auto"
                    dtype = torch.float16
            else:
                device_map = None
                dtype = torch.float32

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,

                torch_dtype=dtype,
                device_map=device_map
            )

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            model = get_peft_model(base_model, lora_config)

            training_args = TrainingArguments(
                output_dir=self.model_path,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                num_train_epochs=1,
                save_strategy="epoch",
                save_total_limit=2,
                fp16=torch.cuda.is_available(),
                logging_dir="./logs",
                logging_steps=10,
                learning_rate=2e-5,
                warmup_ratio=0.03,
                lr_scheduler_type="cosine",
                report_to="none"
            )


            def tokenize_function(examples):
                model_inputs = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": []
                }

                for idx in range(len(examples["input"])):
                    input_text = examples["input"][idx]
                    output_text = examples["output"][idx]

                    inputs = self.tokenizer(
                        input_text,
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )

                    outputs = self.tokenizer(
                        output_text,
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )

                    model_inputs["input_ids"].append(inputs["input_ids"].squeeze().tolist())
                    model_inputs["attention_mask"].append(inputs["attention_mask"].squeeze().tolist())
                    model_inputs["labels"].append(outputs["input_ids"].squeeze().tolist())

                return model_inputs


            tokenized_dataset = training_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=training_dataset.column_names
            )


            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=model,
                return_tensors="pt"
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator
            )

            trainer.train()
            model.save_pretrained(self.model_path)

            logger.info("Fine-tuned model saved successfully")
            return True

        except Exception as e:
            logger.error(f"Error fine-tuning model: {str(e)}")
            return False

    def generate_formulas_and_derivations(self, subject, topic=None):
        try:
            if topic:
                query = f"{subject} {topic} formulas and derivation"
            else:
                query = f"{subject} formulas and derivation"

            relevant_docs = self.vectorstore.similarity_search(query, k=5)
            context = "\n".join([doc.page_content for doc in relevant_docs])


            max_context_length = 6000
            context = context[:max_context_length]

            if not self.generator_ready:
                return {
                    'success': False,
                    'error': "Model not initialized properly",
                    'formulas': [],
                    'derivations': []
                }

            prompt = f"""
             <s>[INST] You are a helpful AI assistant that specializes in education. 

             CONTEXT:
             {context}

             Based on the above context about {subject} {topic if topic else ''}, 
             extract and list the most important formulas. For each formula, provide:
             1. The formula name
             2. The formula itself with proper mathematical notation
             3. A brief explanation of when and how to use it

             After listing the formulas, provide detailed derivations for the 3 most important ones.
             Format your response clearly with appropriate headings. [/INST]</s>
             """
            response = self.pipeline(
                prompt,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7
            )[0]['generated_text']
            response_text = response.split("[/INST]</s>")[-1].strip()

            return {
                'success': True,
                'formulas_and_derivations': response_text
            }
        except Exception as e:
            logger.error(f"Error generating formulas and derivations: {str(e)}")
            return {
                'success': False,
                'error': f"Error generating formulas and derivation: {str(e)}",
                'formulas_and_derivations': ""
            }

    def generate_practice_questions(self, subject, topic=None, difficulty="medium", count=5):
        try:
            if topic:
                query = f"{subject} {topic} questions and solutions"
            else:
                query = f"{subject} questions and solutions"

            relevant_docs = self.vectorstore.similarity_search(query, k=5)
            context = "\n".join([doc.page_content for doc in relevant_docs])


            max_context_length = 6000
            context = context[:max_context_length]

            if not self.generator_ready:
                return {
                    'success': False,
                    'error': "Model not initialized properly",
                    'questions': []
                }

            prompt = f"""
             <s>[INST] You are a helpful AI assistant that specializes in education. 

             CONTEXT:
             {context}

             Based on the above context about {subject} {topic if topic else ''}, 
             generate {count} {difficulty} level practice questions. For each question:
             1. Provide a clear, well-formulated question
             2. Include a step-by-step solution
             3. Highlight the key concepts used in solving the problem

             Format your response as a numbered list of questions, each with its solution.
             Make sure the questions are of {difficulty} difficulty level. [/INST]</s>
             """

            response = self.pipeline(
                prompt,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7
            )[0]['generated_text']
            text_response = response.split("[/INST]</s>")[-1].strip()

            return {
                'success': True,
                'questions_and_solutions': text_response
            }
        except Exception as e:
            logger.error(f"Error generating practice questions: {str(e)}")
            return {
                'success': False,
                'error': f"Error generating practice questions: {str(e)}",
                'questions_and_solutions': ""
            }

    def answer_question(self, question):
        try:
            relevant_docs = self.vectorstore.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in relevant_docs])


            max_context_length = 6000
            context = context[:max_context_length]

            if not self.generator_ready:
                return {
                    'success': False,
                    'error': "Model not initialized properly",
                    'answer': ""
                }

            prompt = f"""
             <s>[INST] You are a helpful AI assistant that specializes in education. 

             CONTEXT:
             {context}

             QUESTION:
             {question}

             Please provide a detailed answer to the question, showing all your work and explanations step by step.
             If a mathematical solution is required, include all steps in the calculation.
             If a derivation is needed, show each step of the derivation clearly.
             If the question requires a conceptual explanation, make sure to be thorough and precise. [/INST]</s>
             """

            response = self.pipeline(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)[0]['generated_text']

            response_text = response.split("[/INST]</s>")[-1].strip()

            return {
                'success': True,
                'answer': response_text
            }
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                'success': False,
                'error': f"Error answering question: {str(e)}",
                'answer': ""
            }