import gradio as gr
import json 

from meeting_minutes_generator import (
    summarize_transcript,
    format_meeting_minutes,
    langfuse,
    prompt,
    openai
)

def generate_minutes_from_text(transcript: str, progress=gr.Progress(track_tqdm=True)):
    """
    Main handler function for the Gradio interface. It takes a transcript, processes it,
    and returns formatted meeting minutes as a Markdown string.
    """
    # Check if the services were initialized correctly
    if not langfuse or not prompt or not openai.api_key:
         return "## Error\nApplication is not configured. Please check your `.env` file and restart."

    if not transcript or not transcript.strip():
        return "## Warning\nPlease provide a transcript in the input box."
    
    try:
        progress(0, desc="Starting Generation...")
        # Call the summarization logic from the other file
        meeting_summary = summarize_transcript(transcript, progress=progress)
        
        progress(0.9, desc="Formatting minutes...")
        # Call the formatting logic from the other file
        formatted_minutes = format_meeting_minutes(meeting_summary)
        
        return formatted_minutes

    except json.JSONDecodeError as je:
        print(f"JSON decoding error: {je}")
        return "## Error\n The AI model returned an invalid format. This can happen with very short or ambiguous text. Please try again with a more detailed transcript."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"## An Unexpected Error Occurred\n`{str(e)}`"
    
if __name__ == "__main__":
    # Load the example transcript to show in the UI
    try:
        with open("transcript.txt", "r", encoding="utf-8") as f:
            example_transcript = f.read()
    except FileNotFoundError:
        example_transcript = "Alice: Let's start. Bob: I agree. Our goal is to finish the project by Friday. Alice: Okay, Bob, you are assigned this task."

    # Define the Gradio interface
    iface = gr.Interface(
        fn=generate_minutes_from_text,
        inputs=gr.Textbox(
            lines=20, 
            label="📝 Meeting Transcript", 
            placeholder="Paste the full meeting transcript here...",
            value=example_transcript
        ),
        outputs=gr.Markdown(
            label="✅ Generated Meeting Minutes",
            show_copy_button=True,
        ),
        title="🤖 AI Meeting Minutes Generator",
        description="""
        Enter a raw meeting transcript to automatically generate structured meeting minutes. 
        The tool will extract the title, date, participants, agenda, discussion points, decisions, and action items.
        It supports multiple languages by translating the text to English before processing.
        """,
        allow_flagging="never",
        examples=[[example_transcript]],
        theme=gr.themes.Soft()
    )

    # Launch the web server
    print("Launching Gradio interface...")
    iface.launch()