from rest_framework import generics
from .models import Project, AboutMe, Skill, Experience, Education
from .serializers import ProjectSerializer, AboutMeSerializer, SkillSerializer, ExperienceSerializer, EducationSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from groq import Groq
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import requests
import numpy as np
from django.core.cache import cache

# Load environment variables
load_dotenv()

# --- Hugging Face API Configuration ---
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-albert-small-v2"
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# --- A better, cached approach for loading the resume ---
def get_resume_chunks():
    """
    Extracts, chunks, and caches the resume text to avoid reading the file on every request.
    """
    # Try to get the chunks from the cache first
    cached_chunks = cache.get('resume_chunks')
    if cached_chunks is not None:
        return cached_chunks

    try:
        # Function to chunk text
        def chunk_text(text, chunk_size=350, overlap=50):
            tokens = text.split()
            chunks = []
            for i in range(0, len(tokens), chunk_size - overlap):
                chunk = " ".join(tokens[i:i + chunk_size])
                chunks.append(chunk)
            return chunks

        # Function to extract text from PDF
        def extract_text_from_pdf(pdf_path):
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text

        # Construct the full path to the resume.pdf file
        RESUME_PDF_PATH = os.path.join(os.path.dirname(__file__), 'resume.pdf')
        resume_text = extract_text_from_pdf(RESUME_PDF_PATH)
        text_chunks = chunk_text(resume_text)
        
        # Store the chunks in the cache for 1 hour
        cache.set('resume_chunks', text_chunks, timeout=3600)
        
        return text_chunks

    except Exception as e:
        print(f"Error processing resume PDF: {e}")
        return []

class IsAdminOrReadOnly(permissions.BasePermission):
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True # Allow GET, HEAD, OPTIONS requests
        return request.user and request.user.is_staff
    
class ChatView(APIView):
    def post(self, request):
        question = request.data.get('question')
        if not question:
            return Response({"error": "Question is required"}, status=status.HTTP_400_BAD_REQUEST)

        # --- 1. Get Resume Chunks ---
        try:
            text_chunks = get_resume_chunks()
            if not text_chunks:
                return Response({"answer": "Sorry, the resume content is not available to answer questions."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            print(f"ERROR reading resume PDF: {e}")
            return Response({"answer": "Sorry, there was an error reading the resume information."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # --- 2. Query Hugging Face for Similarity ---
        try:
            def query_hf_similarity(question, chunks):
                payload = {
                    "inputs": { "source_sentence": question, "sentences": chunks },
                    "options": {"wait_for_model": True}
                }
                response = requests.post(API_URL, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()

            similarity_scores = query_hf_similarity(question, text_chunks)
            most_relevant_chunk_index = np.argmax(similarity_scores)
            relevant_chunk = text_chunks[most_relevant_chunk_index]

        except requests.exceptions.RequestException as e:
            print(f"Hugging Face API Error: {e}")
            return Response({"answer": "Sorry, I'm having trouble connecting to the similarity service right now."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            print(f"ERROR processing similarity scores: {e}")
            return Response({"answer": "Sorry, there was an error analyzing your question's relevance."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


        # --- 3. Get Answer from Groq API ---
        try:
            session_id = request.session.session_key
            if not session_id:
                request.session.create()
                session_id = request.session.session_key

            conversation_history = cache.get(session_id, [])

            groq_prompt = f"""
            You are Abhishek Yadav, a college student with strong technical skills.
            Answer the question below in FIRST PERSON as if you're directly speaking to an HR representative.
            Base your answer *only* on the provided "Background Information" chunk from your resume.
            Keep your answers SHORT, PRECISE, and PROFESSIONAL.

            Background Information:
            "{relevant_chunk}"

            Question: {question}
            Answer:
            """

            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are Abhishek Yadav. Answer questions in FIRST PERSON as a professional. Keep your answers SHORT and PRECISE."},
                    *conversation_history,
                    {"role": "user", "content": groq_prompt}
                ],
                model="openai/gpt-oss-20b", # Switched to a known fast and reliable model
                temperature=0.3,
            )
            
            answer = response.choices[0].message.content

            conversation_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])
            cache.set(session_id, conversation_history, timeout=3600)

            return Response({"answer": answer}, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"ERROR with Groq API or session management: {e}")
            return Response({"answer": "Sorry, the AI chat service is not responding correctly."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --- Your other views remain unchanged ---

class ProjectList(generics.ListCreateAPIView):
    permission_classes = [IsAdminOrReadOnly]
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer

class ProjectDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAdminOrReadOnly]
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer

class AboutMeDetail(generics.RetrieveUpdateAPIView): # Only one AboutMe object, so no create/delete list
    permission_classes = [IsAdminOrReadOnly]
    queryset = AboutMe.objects.all()
    serializer_class = AboutMeSerializer
    def get_object(self):
        return AboutMe.objects.first()

class SkillList(generics.ListCreateAPIView):
    permission_classes = [IsAdminOrReadOnly]
    queryset = Skill.objects.all()
    serializer_class = SkillSerializer

class SkillDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAdminOrReadOnly]
    queryset = Skill.objects.all()
    serializer_class = SkillSerializer

class ExperienceList(generics.ListCreateAPIView):
    permission_classes = [IsAdminOrReadOnly]
    queryset = Experience.objects.all()
    serializer_class = ExperienceSerializer

class ExperienceDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAdminOrReadOnly]
    queryset = Experience.objects.all()
    serializer_class = ExperienceSerializer

class EducationList(generics.ListCreateAPIView):
    permission_classes = [IsAdminOrReadOnly]
    queryset = Education.objects.all()
    serializer_class = EducationSerializer

class EducationDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAdminOrReadOnly]
    queryset = Education.objects.all()
    serializer_class = EducationSerializer