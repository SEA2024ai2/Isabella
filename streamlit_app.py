import streamlit as st
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import logging
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a StringIO object to capture log output
log_stream = io.StringIO()
stream_handler = logging.StreamHandler(log_stream)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Streamlit UI setup
st.title("📧 AI Email Assistant")

# Use st.secrets for sensitive information
openai_api_key = st.secrets["OPENAI_API_KEY"]
email_address = st.secrets["email_address"]
email_password = st.secrets["email_password"]

# Create an OpenAI client
client = OpenAI(api_key=openai_api_key)

# Function to send email
def send_email(to_address, subject, body, cc=None, bcc=None):
    logger.info(f"Preparing to send email to: {to_address}")
    msg = MIMEMultipart()
    msg['From'] = email_address
    msg['To'] = to_address
    msg['Subject'] = subject
    if cc:
        msg['Cc'] = cc
        logger.info(f"CC recipients: {cc}")
    if bcc:
        msg['Bcc'] = bcc
        logger.info(f"BCC recipients: {bcc}")
    msg.attach(MIMEText(body, 'plain'))

    recipients = [to_address]
    if cc:
        recipients.extend(cc.split(','))
    if bcc:
        recipients.extend(bcc.split(','))

    try:
        logger.info("Attempting to connect to SMTP server...")
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            logger.info("Connected to SMTP server")
            server.starttls()
            logger.info("TLS connection established")
            server.login(email_address, email_password)
            logger.info("Logged in to email account")
            server.send_message(msg)
            logger.info("Email sent successfully")
        return "Email sent successfully!"
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}", exc_info=True)
        return f"Failed to send email: {str(e)}"

# Define the function for the AI to use
functions = [
    {
        "name": "compose_email",
        "description": "Compose an email based on user instructions",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient's email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body content"},
                "cc": {"type": "string", "description": "CC recipients, comma-separated"},
                "bcc": {"type": "string", "description": "BCC recipients, comma-separated"},
            },
            "required": ["to", "subject", "body"]
        }
    }
]

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []
if "email_to_send" not in st.session_state:
    st.session_state.email_to_send = None
if "waiting_for_confirmation" not in st.session_state:
    st.session_state.waiting_for_confirmation = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help with your email?"):
    logger.info(f"Received user input: {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.waiting_for_confirmation:
        if any(phrase in prompt.lower() for phrase in ["yes", "send it", "confirm", "go ahead"]):
            if st.session_state.email_to_send:
                result = send_email(
                    st.session_state.email_to_send['to'],
                    st.session_state.email_to_send['subject'],
                    st.session_state.email_to_send['body'],
                    st.session_state.email_to_send.get('cc'),
                    st.session_state.email_to_send.get('bcc')
                )
                logger.info(f"Email sending result: {result}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Email sent: {result}"
                })
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.messages[-1]["content"])
                st.session_state.email_to_send = None
                st.session_state.waiting_for_confirmation = False
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I'm sorry, but I don't have an email ready to send. Could you please provide the details for a new email?"
                })
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.messages[-1]["content"])
                st.session_state.waiting_for_confirmation = False
        else:
            st.session_state.waiting_for_confirmation = False
            # Continue with normal conversation flow
    
    if not st.session_state.waiting_for_confirmation:
        # Generate a response using OpenAI with function calling
        logger.info("Sending request to OpenAI API")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            functions=functions,
            function_call="auto"
        )
        logger.info("Received response from OpenAI API")

        assistant_message = response.choices[0].message
        
        # Check if the model wants to call a function
        if assistant_message.function_call:
            logger.info(f"AI decided to call function: {assistant_message.function_call.name}")
            function_name = assistant_message.function_call.name
            function_args = json.loads(assistant_message.function_call.arguments)
            
            if function_name == "compose_email":
                # Log the email details
                logger.info(f"Email Details - To: {function_args['to']}, Subject: {function_args['subject']}")
                logger.info(f"Email Body: {function_args['body'][:100]}...")  # Log first 100 chars of body
                if 'cc' in function_args:
                    logger.info(f"CC: {function_args['cc']}")
                if 'bcc' in function_args:
                    logger.info(f"BCC: {function_args['bcc']}")
                
                # Store the email details for confirmation
                st.session_state.email_to_send = function_args
                
                # Present the email for confirmation
                confirmation_message = f"""I've composed an email based on your request. Here are the details:

To: {function_args['to']}
Subject: {function_args['subject']}
Body: {function_args['body']}

Would you like me to send this email? Please confirm by saying 'Yes' or 'Send it'."""
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": confirmation_message
                })
                st.session_state.waiting_for_confirmation = True
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })
        else:
            # If no function call, just display the response
            logger.info("AI response did not include a function call")
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_message.content
            })
        
        with st.chat_message("assistant"):
            st.markdown(st.session_state.messages[-1]["content"])

# Display logs in Streamlit
st.subheader("Logs")
st.text_area("Detailed Logs", log_stream.getvalue(), height=300)