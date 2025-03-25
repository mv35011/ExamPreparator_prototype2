from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import json
import logging
from dotenv import load_dotenv
import os
import time

logger = logging.getLogger(__name__)
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


class GenerateTrainingData:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.2,
            groq_api_key=groq_api_key,
            model_name="deepseek-r1-distill-llama-70b-specdec"
        )

    def generate_training_data(self, input_text=None):

        try:
            subjects = {
                "Mathematics": ["Laplace Transform", "Fourier Series", "Probability", "Calculus",
                                "Linear Algebra", "Differential Equations", "Vector Calculus", "Complex Analysis"],
                "Physics": ["Quantum Mechanics", "Electromagnetic Waves", "Thermodynamics",
                            "Classical Mechanics", "Relativity", "Optics", "Nuclear Physics"],
                "Computer Science": ["Sorting Algorithms", "Recursion", "Graph Theory",
                                     "Dynamic Programming", "Data Structures", "Machine Learning Fundamentals",
                                     "Computational Complexity"]
            }

            training_data = []

            context_template = PromptTemplate.from_template(
                """You are an expert educator creating content for a technical education system.

                Topic: {subject} - {topic}

                Please provide:
                1. A comprehensive explanation of this topic (2-3 paragraphs)
                2. Key formulas and theorems related to this topic
                3. Common applications of this topic

                {additional_context}

                Format your response as a structured educational text.
                """
            )

            qa_template = PromptTemplate.from_template(
                """Based on the following educational content about {subject} - {topic}, create:

                1. A challenging but solvable problem related to this topic
                2. A detailed step-by-step derivation showing how to solve the problem

                Content:
                {context}

                Format your response as:

                QUESTION:
                [Write a clearly stated problem here]

                DERIVATION:
                [Provide a detailed step-by-step solution with clear explanations of each step]
                """
            )

            for subject, topics in subjects.items():
                logger.info(f"Generating training data for {subject}")
                for topic in topics:
                    try:
                        logger.info(f"Processing topic: {topic}")


                        additional_context = input_text if input_text else ""
                        context_response = self.llm.invoke(
                            context_template.format(
                                subject=subject,
                                topic=topic,
                                additional_context=additional_context
                            )
                        )

                        context_text = context_response.content


                        qa_response = self.llm.invoke(
                            qa_template.format(
                                subject=subject,
                                topic=topic,
                                context=context_text
                            )
                        )

                        qa_text = qa_response.content


                        try:
                            question = ""
                            derivation = ""

                            if "QUESTION:" in qa_text and "DERIVATION:" in qa_text:
                                parts = qa_text.split("DERIVATION:")
                                question_part = parts[0]
                                derivation = parts[1].strip()

                                question = question_part.split("QUESTION:")[1].strip()
                            else:

                                lines = qa_text.split('\n')
                                question_started = False
                                derivation_started = False

                                for line in lines:
                                    if "question" in line.lower() or "problem" in line.lower():
                                        question_started = True
                                        derivation_started = False
                                        continue
                                    elif "solution" in line.lower() or "step" in line.lower() or "derivation" in line.lower():
                                        derivation_started = True
                                        question_started = False
                                        continue

                                    if question_started:
                                        question += line + "\n"
                                    elif derivation_started:
                                        derivation += line + "\n"


                            training_example = {
                                "context": context_text,
                                "question": question.strip(),
                                "derivation": derivation.strip()
                            }


                            if len(training_example["question"]) > 10 and len(training_example["derivation"]) > 20:
                                training_data.append(training_example)
                                logger.info(f"Added training example for {subject} - {topic}")
                            else:
                                logger.warning(f"Skipping invalid training example for {subject} - {topic}")

                        except Exception as e:
                            logger.error(f"Error parsing response for {topic}: {str(e)}")


                        time.sleep(1)

                    except Exception as e:
                        logger.error(f"Error generating data for {topic}: {str(e)}")


            if training_data:
                with open("fine_tuning_data.json", "w") as f:
                    json.dump(training_data, f, indent=4)

                logger.info(f"Successfully generated and saved {len(training_data)} training examples")
                return True
            else:
                logger.error("No valid training data was generated")
                return False

        except Exception as e:
            logger.error(f"Error in generate_training_data: {str(e)}")
            return False

    def generate_advanced_examples(self, num_examples=20):

        try:
            problem_types = [
                "Proof problems requiring mathematical derivation",
                "Multi-step calculation problems with intermediate steps",
                "Application problems connecting theory to real-world scenarios",
                "Conceptual explanation problems requiring detailed reasoning",
                "Problems requiring formula derivation from first principles"
            ]

            difficulty_levels = ["medium", "hard"]

            advanced_template = PromptTemplate.from_template(
                """You are creating specialized training data for an AI that helps students with exams.

                Problem type: {problem_type}
                Difficulty: {difficulty}
                Subject area: {subject_area}

                Create a detailed educational example with:

                1. A rich context paragraph explaining relevant concepts (2-3 paragraphs)
                2. A challenging but solvable problem of the specified type
                3. A step-by-step derivation/solution showing detailed reasoning

                Format your response as:

                CONTEXT:
                [Educational context here]

                QUESTION:
                [Problem statement here]

                DERIVATION:
                [Detailed step-by-step solution]
                """
            )

            training_data = []

            for i in range(num_examples):
                subject_area = f"{['Mathematics', 'Physics', 'Computer Science'][i % 3]} - {['Advanced Topic', 'Theoretical Foundations', 'Applied Methods'][i % 3]}"
                problem_type = problem_types[i % len(problem_types)]
                difficulty = difficulty_levels[i % len(difficulty_levels)]

                try:
                    response = self.llm.invoke(
                        advanced_template.format(
                            problem_type=problem_type,
                            difficulty=difficulty,
                            subject_area=subject_area
                        )
                    )

                    text = response.content

                    # Parse the response
                    context = ""
                    question = ""
                    derivation = ""

                    if "CONTEXT:" in text and "QUESTION:" in text and "DERIVATION:" in text:
                        parts = text.split("QUESTION:")
                        context = parts[0].replace("CONTEXT:", "").strip()

                        qd_parts = parts[1].split("DERIVATION:")
                        question = qd_parts[0].strip()
                        derivation = qd_parts[1].strip()

                        # Add to training data
                        training_example = {
                            "context": context,
                            "question": question,
                            "derivation": derivation
                        }


                        if len(context) > 100 and len(question) > 10 and len(derivation) > 100:
                            training_data.append(training_example)
                            logger.info(f"Added advanced training example #{i + 1}")
                        else:
                            logger.warning(f"Skipping invalid advanced example #{i + 1}")
                    else:
                        logger.warning(f"Response format incorrect for example #{i + 1}")

                except Exception as e:
                    logger.error(f"Error generating advanced example #{i + 1}: {str(e)}")

                #
                time.sleep(2)


            try:
                existing_data = []
                if os.path.exists("fine_tuning_data.json"):
                    with open("fine_tuning_data.json", "r") as f:
                        existing_data = json.load(f)

                combined_data = existing_data + training_data

                with open("fine_tuning_data.json", "w") as f:
                    json.dump(combined_data, f, indent=4)

                logger.info(f"Successfully added {len(training_data)} advanced examples to training data")
                return True
            except Exception as e:
                logger.error(f"Error saving advanced examples: {str(e)}")


                with open("advanced_training_data.json", "w") as f:
                    json.dump(training_data, f, indent=4)
                logger.info(f"Saved {len(training_data)} advanced examples to separate file")
                return True

        except Exception as e:
            logger.error(f"Error in generate_advanced_examples: {str(e)}")
            return False

    def validate_and_clean_training_data(self, input_file="fine_tuning_data.json",
                                         output_file="cleaned_training_data.json"):

        try:
            if not os.path.exists(input_file):
                logger.error(f"Input file {input_file} does not exist")
                return False

            with open(input_file, "r") as f:
                data = json.load(f)

            logger.info(f"Loaded {len(data)} training examples for validation")

            cleaned_data = []
            for i, example in enumerate(data):
                try:

                    if not all(key in example for key in ["context", "question", "derivation"]):
                        logger.warning(f"Example #{i} missing required fields")
                        continue


                    if len(example["context"]) < 50:
                        logger.warning(f"Example #{i} has too short context")
                        continue

                    if len(example["question"]) < 10:
                        logger.warning(f"Example #{i} has too short question")
                        continue

                    if len(example["derivation"]) < 50:
                        logger.warning(f"Example #{i} has too short derivation")
                        continue


                    derivation = example["derivation"].lower()
                    has_steps = any(marker in derivation for marker in ["step", "first", "next", "then", "finally"])
                    has_math = any(
                        marker in derivation for marker in ["=", "+", "-", "×", "∫", "∂", "formula", "equation"])

                    if not (has_steps or has_math):
                        logger.warning(f"Example #{i} lacks step markers or math content")
                        continue


                    cleaned_data.append(example)

                except Exception as e:
                    logger.error(f"Error validating example #{i}: {str(e)}")

            logger.info(f"Validation complete. Kept {len(cleaned_data)} of {len(data)} examples")

            with open(output_file, "w") as f:
                json.dump(cleaned_data, f, indent=4)

            logger.info(f"Cleaned data saved to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error in validate_and_clean_training_data: {str(e)}")
            return False