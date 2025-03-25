import os
import tempfile
from pathlib import Path


class StreamlitFileAdapter:
    """
    Adapter class that mimics Django's FileSystemStorage for use with Streamlit.
    This provides compatibility with the AIExamHelper class that was designed for Django.
    """

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.files = {}

    def save(self, name, content):
        """
        Save uploaded file content to a temporary file

        Args:
            name: The name of the file
            content: File-like object or path to file

        Returns:
            The name of the file saved
        """
        # Generate a unique filename to avoid collisions
        unique_name = f"{Path(name).stem}_{os.urandom(8).hex()}{Path(name).suffix}"
        file_path = os.path.join(self.temp_dir, unique_name)

        # If content is a path string, copy the file
        if isinstance(content, str) and os.path.exists(content):
            with open(content, 'rb') as src, open(file_path, 'wb') as dst:
                dst.write(src.read())
        # If content has a 'read' method (file-like object)
        elif hasattr(content, 'read'):
            with open(file_path, 'wb') as f:
                f.write(content.read())
        # Otherwise, try to write the content directly
        else:
            with open(file_path, 'wb') as f:
                f.write(content)

        # Store the mapping
        self.files[unique_name] = file_path

        return unique_name

    def path(self, name):
        """
        Return the full path to the file

        Args:
            name: The name of the file (as returned by save)

        Returns:
            The full path to the file
        """
        return self.files.get(name)

    def delete(self, name):
        """
        Delete the file

        Args:
            name: The name of the file (as returned by save)

        Returns:
            True if the file was deleted successfully
        """
        if name in self.files:
            try:
                os.remove(self.files[name])
                del self.files[name]
                return True
            except Exception:
                return False
        return False

    def cleanup(self):
        """Clean up all temporary files"""
        for name in list(self.files.keys()):
            self.delete(name)

        try:
            os.rmdir(self.temp_dir)
        except Exception:
            pass