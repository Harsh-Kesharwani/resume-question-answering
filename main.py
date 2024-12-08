import os
import gradio as gr
import fitz  # PyMuPDF for handling PDF files
import docx  # python-docx for handling Word documents
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from faster_whisper import WhisperModel
from question_answering import QuestionAnsweringAgent

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

class AudioTranscriber:
    """
    Handles audio transcription using different methods.
    """
    def __init__(self, model_path="base"):
        """
        Initialize the transcriber with Whisper models.
        
        :param model_path: Path to the Whisper model
        """
        load_dotenv()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.whisper_model = WhisperModel(model_path, device="cpu", compute_type="int8")

    def transcribe_audio(self, audio_input):
        """
        Transcribes audio input into text using the Faster Whisper model.
        
        :param audio_input: Path to the audio file
        :return: Transcribed text or error message
        """
        try:
            # Load and preprocess the audio file
            segments, _ = self.whisper_model.transcribe(audio_input, beam_size=5)
            # Combine segments into a single transcription text
            transcription = " ".join(segment.text for segment in segments)
            print(f"Transcription: {transcription}")  # Log the transcription
            return transcription
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return f"Error during transcription: {str(e)}"



class ResumeParser:
    """
    Handles parsing of different resume file formats.
    """
    @staticmethod
    def parse_resume(resume_file):
        """
        Converts uploaded resume file into plain text and logs to console.
        Automatically triggered upon file upload.
        """
        if resume_file:
            try:
                # If the file is a PDF
                if resume_file.name.endswith(".pdf"):
                    doc = fitz.open(resume_file.name)
                    resume_content = ""
                    for page in doc:
                        resume_content += page.get_text("text")
                    print("Resume Content (PDF):\n", resume_content)  # Log resume content to console
                    return resume_content
                # If the file is a Word document
                elif resume_file.name.endswith(".docx"):
                    doc = docx.Document(resume_file.name)
                    resume_content = "\n".join([para.text for para in doc.paragraphs])
                    print("Resume Content (DOCX):\n", resume_content)  # Log resume content to console
                    return resume_content
                # If the file is plain text
                elif resume_file.name.endswith(".txt"):
                    with open(resume_file.name, "r", encoding="utf-8") as f:
                        resume_content = f.read()
                        print("Resume Content (TXT):\n", resume_content)  # Log resume content to console
                    return resume_content
                else:
                    print(f"Unsupported file type: {resume_file.name}")
            except Exception as e:
                print(f"Error reading resume: {str(e)}")
                return f"Error reading resume: {str(e)}"
        return None

class ResumeQAApp:
    """
    Main application class for Resume Question Answering.
    """
    def __init__(self):
        """
        Initialize the application components.
        """
        self.audio_transcriber = AudioTranscriber()
        self.resume_parser = ResumeParser()
        self.qa_agent = None
        self.setup_ui()

    def handle_file_upload(self, resume_file):
        """
        Handles resume file upload and initializes QA agent.
        
        :param resume_file: Uploaded resume file
        :return: Upload status message
        """
        if not resume_file:
            print("No resume file provided")
            return "No file uploaded"

        try:
            resume_content = self.resume_parser.parse_resume(resume_file)
            
            if resume_content:
                print(f"Successfully parsed resume. Content length: {len(resume_content)} characters")
                self.qa_agent = QuestionAnsweringAgent(resume_content)
                return f"Successfully uploaded. Content length: {len(resume_content)} characters"
            else:
                print(f"Failed to parse resume.")
                self.qa_agent = None
                return "Failed to parse resume file"
        
        except Exception as e:
            print(f"Error in file upload: {e}")
            return f"Error processing file: {str(e)}"

    def generate_response(self, transcript, resume_file):
        """
        Generate response by parsing resume and answering question.
        
        :param transcript: Transcribed audio text
        :param resume_file: Uploaded resume file
        :return: Tuple of (identified question, answer)
        """
        if not self.qa_agent:
            return "No resume uploaded or parsing failed.", "N/A"
        
        return self.qa_agent.answer_question(transcript)

    def setup_ui(self):
        """
        Set up the Gradio UI for the application.
        """
        with gr.Blocks() as self.demo:
            gr.Markdown("# 📄 Resume Question Answering AI Demo")

            with gr.Row():
                # with gr.Column():
                #     # Audio Input and Transcription
                #     self.audio_input = gr.Audio(type="filepath", label="Speak (Audio Input)")
                #     self.transcription = gr.Textbox(label="Transcription")
                #     self.transcribe_button = gr.Button("Transcribe Audio")
                #     self.transcribe_button.click(
                #         self.audio_transcriber.transcribe_audio, 
                #         inputs=self.audio_input, 
                #         outputs=self.transcription
                #     )
                
                with gr.Column():
                    # Audio Input and Real-Time Transcription
                    self.audio_input = gr.Audio(type="filepath", label="Speak (Audio Input)")
                    self.transcription = gr.Textbox(label="Transcription")
                    
                    # Enable real-time transcription by linking audio input to the transcription function
                    self.audio_input.change(
                        self.audio_transcriber.transcribe_audio,
                        inputs=self.audio_input,
                        outputs=self.transcription
                    )

                with gr.Column():
                    # Resume Upload and Response Generation
                    self.resume_file = gr.File(label="Upload Resume", file_types=[".pdf", ".docx", ".txt"])
                    self.upload_status = gr.Textbox(label="Upload Status", interactive=False)

                    self.resume_file.upload(
                        self.handle_file_upload, 
                        inputs=[self.resume_file],
                        outputs=[self.upload_status]
                    )

                    self.generate_button = gr.Button("Generate Response")
                    self.question_output = gr.Textbox(label="Question Identified")
                    self.answer_output = gr.Textbox(label="Answer")

                    # Bind generate response method
                    self.generate_button.click(
                        self.generate_response, 
                        inputs=[self.transcription, self.resume_file], 
                        outputs=[self.question_output, self.answer_output]
                    )

    def launch(self):
        """
        Launch the Gradio application.
        """
        global demo  
        demo = self.demo
        self.demo.launch()
        

if __name__ == "__main__":
    app = ResumeQAApp()
    app.launch()
