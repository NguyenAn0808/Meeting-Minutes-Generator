import os
import json
import re
import tiktoken

from langfuse.openai import openai
from langfuse import Langfuse
from langfuse.decorators import observe
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Initialize Langfuse with provided API keys and host from environment variables
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    environment="production"
)

openai.api_key = os.getenv("OPENAI_API_KEY")

#Meeting Mintues template output
def format_meeting_minutes(meeting_data: dict) -> str:
    """
    Convert structured meeting data (JSON) into a formatted meeting minutes document.
    """
    title = meeting_data.get("meeting_title", "Meeting Minutes")
    meeting_date = meeting_data.get("meeting_date", "To Be Decided")
    meeting_time = meeting_data.get("meeting_time", "To Be Decided")
    meeting_mode = meeting_data.get("meeting_mode", "Unknown")
    meeting_place = meeting_data.get("meeting_place", "Unknown")
    # Gộp thông tin mode với place nếu mode có giá trị khác "Unknown"
    if meeting_mode and meeting_mode != "Unknown":
        place_info = f"{meeting_place} ({meeting_mode})"
    else:
        place_info = meeting_place

    participants = ', '.join(meeting_data.get("participants", [])) or "No participants listed."
    agenda = "\n".join(f"- {item}" for item in meeting_data.get("agenda", [])) or "No agenda provided."
    
    # Minutes: discussion_points, decisions_made, and action_items
    minutes = meeting_data.get("minutes", {})
    discussion_points = "\n".join(f"- {point}" for point in minutes.get("discussion_points", [])) or "No discussion points recorded."
    decisions_made = "\n".join(f"- {decision}" for decision in minutes.get("decisions_made", [])) or "No decisions recorded."
    action_items = "\n".join(
        f"- **{item.get('assignee', 'Unknown')}**: {item.get('task', 'No task provided')} (⏳ Deadline: {item.get('deadline', 'Not specified')})"
        for item in minutes.get("action_items", [])
    ) or "No action items assigned."

    next_meeting = meeting_data.get("next_meeting", "To Be Decided")
    notes = "\n".join(f"- {note}" for note in meeting_data.get("notes", [])) or "No additional remarks."

    template = f"""
    📝 **Meeting Minutes: {title}**

    **📅 Date:** {meeting_date}
    **🕒 Time:** {meeting_time}
    **📍 Location:** {place_info}
    **👥 Participants:** {participants}

    ---
    ## 📝 Agenda  
    {agenda}

    ---
    ## 🔑 Discussion Points  
    {discussion_points}

    ---
    ## ✅ Decisions Made  
    {decisions_made}

    ---
    ## 📋 Action Items  
    {action_items}

    ---
    ## 🗒 Additional Notes  
    {notes}

    ---
    ## 📅 Next Meeting  
    {next_meeting}
    ---
    """
    return template.strip()

# Create the prompt in Langfuse
langfuse.create_prompt(
    name="meeting_minute_generation_v4",
    prompt="Extract the key information from this text and return it in JSON format. Use the following schema: {{json_schema}}",
    config={
        "model": "gpt-4-turbo",
        "temperature": 0.1, # Just summary no create more information
        "json_schema": {
        "type": "object",
        "properties": {
            "meeting_title": {"type": "string"},
            "meeting_date": {"type": "string"},
            "meeting_time": {"type": "string"},
            "meeting_mode": {"type": "string"},
            "meeting_place": {"type": "string"},
            "participants": {"type": "array", "items": {"type": "string"}},
            "agenda": {"type": "array", "items": {"type": "string"}},
            "minutes": {
                "type": "object",
                "properties": {
                    "discussion_points": {"type": "array", "items": {"type": "string"}},
                    "decisions_made": {"type": "array", "items": {"type": "string"}},
                    "action_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},
                                "assignee": {"type": "string"},
                                "deadline": {"type": "string"}
                            },
                            "required": ["task", "assignee", "deadline"]
                        }
                    }
                }
            },
            "next_meeting": {"type": "string"},
            "notes": {"type": "array", "items": {"type": "string"}}
        }
    }
    },
    labels=["production"]
)

# Retrieve the prompt from Langfuse
prompt = langfuse.get_prompt("meeting_minute_generation_v4")

# Function to detect language
def detect_language(text: str) -> str:
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Detect the language of the following text and return only the language code (e.g., 'en' for English, 'vi' for Vietnamese)."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "unknown"

def translate_to_english(text: str) -> str:
    language =  detect_language(text)
    if language == "en":
        return text  # Skip translation if already in English

    print(f"Detected language: {language}. Translating to English...")

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=2000,  
        messages=[
            {"role": "system", "content": "Translate the following text to English in a formal style."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()

tokenizer = tiktoken.encoding_for_model("gpt-4-turbo")
# Function to split long transcripts into smaller chunks
def split_transcript(transcript: str, max_tokens: int = 2000) -> list:
    """
    Split transcript into smaller chunks based on sentence boundaries,
    ensuring each chunk stays within max_tokens.
    """
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        temp_chunk = current_chunk + " " + sentence
        if len(tokenizer.encode(temp_chunk)) > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = temp_chunk
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

@observe()
def summarize_chunk(chunk: str) -> dict:
    """Summarize a chunk of transcript into structured JSON format."""
    prompt = langfuse.get_prompt("meeting_minute_generation_v4")
    json_schema = json.dumps(prompt.config["json_schema"], indent=2)
    system_message = prompt.compile(json_schema=json_schema)
    response = openai.chat.completions.create(
        model=prompt.config["model"],
        temperature=prompt.config["temperature"],
        max_tokens=800,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": chunk}
        ]
    )
    return json.loads(response.choices[0].message.content)

def batch_chunks(chunks, batch_size=3000):
    """Grouping chunks to reduce the number of API calls."""
    batched_chunks = []
    temp_batch = ""
    for chunk in chunks:
        if len(tokenizer.encode(temp_batch + " " + chunk)) > batch_size:
            batched_chunks.append(temp_batch.strip())
            temp_batch = chunk
        else:
            temp_batch += " " + chunk
    if temp_batch:
        batched_chunks.append(temp_batch.strip())
    return batched_chunks

def summarize_batched_chunks(batched_chunks: list) -> list:
    """Summarize multiple transcripts in parallel"""
    with ThreadPoolExecutor(max_workers=min(5, len(batched_chunks))) as executor:
        return list(executor.map(summarize_chunk, batched_chunks))
    
@observe()
def summarize_transcript(transcript: str) -> dict:
    """Summarize an entire transcript by processing chunks asynchronously."""
    transcript =  translate_to_english(transcript)
    chunks = split_transcript(transcript)

    batched_chunks = batch_chunks(chunks)
    summaries = summarize_batched_chunks(batched_chunks)
    
    final_summary = {
        "meeting_title": "Unknown",  # Giá trị mặc định nếu không có trích xuất
        "meeting_date": "Unknown",
        "meeting_time": "Unknown",
        "meeting_place": "Unknown",
        "meeting_mode": "Unknown",
        "participants": [],
        "agenda": [],
        "minutes": {
            "discussion_points": [],
            "decisions_made": [],
            "action_items": []
        },
        "next_meeting": "To Be Decided",
        "notes": []
    }

    for s in summaries:
        if s.get("meeting_title"):
            final_summary["meeting_title"] = s.get("meeting_title")
        if s.get("meeting_date"):
            final_summary["meeting_date"] = s.get("meeting_date")
        if s.get("meeting_time"):
            final_summary["meeting_time"] = s.get("meeting_time")
        if s.get("meeting_place"):
            final_summary["meeting_place"] = s.get("meeting_place")
        if s.get("meeting_mode"):
            final_summary["meeting_mode"] = s.get("meeting_mode")
    
        if s.get("participants"):
            final_summary["participants"] = list(set(final_summary["participants"] + s.get("participants")))
        if s.get("agenda"):
            final_summary["agenda"] = list(set(final_summary["agenda"] + s.get("agenda")))
        
        minutes = s.get("minutes", {})
        if minutes.get("discussion_points"):
            final_summary["minutes"]["discussion_points"] += minutes.get("discussion_points")
        if minutes.get("decisions_made"):
            final_summary["minutes"]["decisions_made"] += minutes.get("decisions_made")
        if minutes.get("action_items"):
            final_summary["minutes"]["action_items"] += minutes.get("action_items")
        
        if s.get("next_meeting"):
            final_summary["next_meeting"] = s.get("next_meeting")
        if s.get("notes"):
            final_summary["notes"] += s.get("notes") 

    return final_summary

def main():
    """
    Main function to read the transcript, summarize it, and save the meeting minutes.
    """
    try:
        # Read the transcript from a file (e.g., 'transcript.txt')
        with open("transcript.txt", "r", encoding="utf-8") as file:
            transcript = file.read()

        if not transcript:
            raise ValueError("Transcript file is empty!")

        # Summarize the transcript
        meeting_summary = summarize_transcript(transcript)

        formatted_meeting_minutes = format_meeting_minutes(meeting_summary) 

        # Save the meeting summary to a file
        with open("meeting_minutes.txt", "w", encoding="utf-8") as file:
            file.write(formatted_meeting_minutes)

        print("Meeting minutes generated successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()