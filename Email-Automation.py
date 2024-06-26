import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
import time
import re

def send_quotation_request(item, grade, quantity, dealer_emails):
    # Email configuration
    sender_email = "anakinlomar@gmail.com"
    subject = f"Quotation Request for Item {item} - Grade {grade} - Quantity {quantity}"

    # Compose the email body
    body = f"Dear Sir/Ma'am,\n\nThis email is to request a quotation for the following :\n\nItem: {item}\nGrade/Type: {grade}\nQuantity: {quantity}\n\nPlease provide your best quotation at your earliest convenience.\n\nBest Regards,\nCompany A"

    # Send the email to each dealer
    for dealer_email in dealer_emails:
        # Set up the MIME
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = dealer_email
        message["Subject"] = subject        
        message.attach(MIMEText(body, "plain"))

        # Connect to the SMTP server (in this case, using Gmail)
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
        
            # Login to your Gmail account
            server.login(sender_email, "iwexinxvgibwpeda")
            server.sendmail(sender_email, dealer_email, message.as_string())

    print("Emails sent successfully!")

def read_email_replies(item, grade, quantity):
    # Email configuration
    username = "anakinlomar@gmail.com"
    password = "iwexinxvgibwpeda"  # Replace with your actual password

    while True:
        # Connect to the Gmail IMAP server
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)
        mail.select("inbox")  # You can select a different mailbox if needed

        # Search for emails with the specified subject
        subject = f"Quotation Request for Item {item} - Grade {grade} - Quantity {quantity}"
        result, data = mail.search(None, '(UNSEEN SUBJECT "{}")'.format(subject))  # Only search for unseen emails

        # Get the list of email IDs
        email_ids = data[0].split()

        # If no unseen emails found, wait for a while and check again
        if not email_ids:
            print("No replies yet. Waiting for replies...")
            mail.close()
            mail.logout()
            time.sleep(30)  # Wait for 30 seconds before checking again
            continue

        # Loop through the email IDs and fetch the emails
        for email_id in email_ids:
            result, msg_data = mail.fetch(email_id, "(RFC822)")
            raw_email = msg_data[0][1]

            # Parse the raw email data
            msg = email.message_from_bytes(raw_email)

            # Display sender and subject
            sender, _ = decode_header(msg.get("From"))[0]
            subject, _ = decode_header(msg.get("Subject"))[0]
            date,_ = decode_header(msg.get("Date"))[0]
            print(f"\nFrom: {sender}\nSubject: {subject}\nDate: {date}\n")

            # Display email body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        body_list = [word.group(1) for word in re.finditer("([\w]*)",body)] #(On (Mon|Tue|Wed|Thu|Fri|Sat|Sun)\, \d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}\, \d{2}[:]\d{2})([.]*)
                        place = []
                        loop = 0
                        for index in body_list:
                            loop += 1
                            if index == 'On':
                                place.append(loop-1)
                            if index in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']:
                                place.append(loop-1)
                        if (place[-1] - place[-2]) == 2:
                                print(" ".join(body_list[0:place[-2]]))


            else:
                print(msg.get_payload(decodAle=True).decode())

            # Mark the email as read
            mail.store(email_id, '+FLAGS', '\Seen')

        mail.close()
        mail.logout()
        break

# User input
item = input("Enter the item: ")
grade = input("Enter the grade/type: ")
quantity = input("Enter the quantity: ")
dealer_emails = input("Enter dealer emails separated by commas: ").split(',')

# Send the quotation request email
send_quotation_request(item, grade, quantity, dealer_emails)

# Wait for and read email replies
read_email_replies(item, grade, quantity)
