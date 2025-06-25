import os
import email
from email import policy
from email.parser import BytesParser

def extract_eml_content(eml_path):
    with open(eml_path, 'rb') as fp:
        msg = BytesParser(policy=policy.default).parse(fp)
        subject = msg['subject']
        sender = msg['from']
        body = msg.get_body(preferencelist=('plain'))
        content = body.get_content() if body else ""
        return {
            "subject": subject,
            "from": sender,
            "content": content.strip()
        }

def load_mails_from_folder(folder_path="data/mails"):
    mails = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".eml"):
            path = os.path.join(folder_path, filename)
            mail = extract_eml_content(path)
            mail["filename"] = filename
            mails.append(mail)
    return mails
