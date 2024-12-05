from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from rest_framework.decorators import api_view
from llama_index.llms.groq import Groq
from llama_index.core import Settings, PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline as QP, InputComponent
from dotenv import load_dotenv
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny

# Load environment variables
load_dotenv()

# Initialize Groq LLM with LlamaIndex
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = Groq(model="Llama3-70b-8192", api_key=GROQ_API_KEY)
Settings.llm = llm

# Define your prompt for finance and crypto queries
response_synthesis_prompt_str = (
    "You are a Smart customer service ai that interacts with human"
    "User Input: {query_str}\n"
    "Response: "
)


response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

# Define the query pipeline
qp = QP(
    modules={
        "input": InputComponent(),
        "response_synthesis_prompt": response_synthesis_prompt,
        "response_synthesis_llm": llm,
    },
    verbose=True,
)

qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

# Define keywords for finance and crypto

import logging

# logger = logging.getLogger(name)
@api_view(['POST'])
@csrf_exempt
@permission_classes([AllowAny])
def agent_query(request):

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query = data.get('query', '')
            if not query:
                return JsonResponse({'error': 'No query provided'}, status=400)
          

            # Check if query contains finance or crypto keywords
         

            # Run the query through LlamaIndex with Groq
            response = qp.run(query=query)
            return JsonResponse({'response': str(response)}, safe=False)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid HTTP method'}, status=405)