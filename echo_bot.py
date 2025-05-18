from flask import Flask, request, jsonify
from pynani import Messenger

PAGE_ACCESS_TOKEN = 'EAAJWI0ApwEkBO7b7KoZCUW8ctIFExPG1CTiZBclZBiQ5CgvEtsuuF12BAU6SvmypNn3QwTA32eewAZAl3MpbFQzCegGQDsHEfw3Ppuy3crJWvEeZBsYnQWPVOcQ7rBolsXy6eEtnSZB1w3P9sNFEiKRIZBo4PPBY52L9ZBBmB3qmoBaZBbUZByFDZAvnBVTLHhGKhk9DwZDZD'
TOKEN = "abc123"

mess = Messenger(PAGE_ACCESS_TOKEN)
app = Flask(__name__)

@app.get("/")
def meta_verify():
    return mess.verify_token(request.args, TOKEN)

@app.post("/")
def meta_webhook():
    data = request.get_json()
    sender_id = mess.get_sender_id(data)
    message = mess.get_message_text(data)
    if message == "Hello":
        mess.send_text_message(sender_id, "Hello, World!")
    if message == "Bye":
        mess.send_text_message(sender_id, "Nice to meet you! 👍🏽")
    if message == "Beabadoobee":
        mess.send_text_message(sender_id, "Beabaduday")

    return jsonify({"status": "success"}), 200

if __name__ =='__main__':
    app.run(port=8080, debug=True)