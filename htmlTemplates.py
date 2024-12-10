import clipboard

css = '''
<style>
.chat-message {
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    display: flex;
    position: relative;
    max-width: 80%;  /* Restrict message width for better alignment */
    line-height: 1.5; /* Improved line height for readability */
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);  /* Subtle shadow for depth */
}

.chat-message.user {
    background-color: #2C3E50;  /* Slightly darker shade of blue */
    color: #ECF0F1; /* Light text color */
}

.chat-message.bot {
    background-color: #34495E;  /* A more muted, professional shade of blue */
    color: #ECF0F1; /* Light text color */
}

.chat-message .avatar {
    width: 15%;
    margin-right: 1rem;
}

.chat-message .avatar img {
    max-width: 60px;
    max-height: 60px;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid #ECF0F1;  /* White border around avatar */
}

.chat-message .message {
    width: 85%;
    padding: 0.8rem 1.5rem;
    font-size: 17px;
    font-family: 'Arial', sans-serif;  /* Modern font */
    background-color: transparent; /* No background inside message */
}

.small-font {
    font-size: 14px !important;
    color: #BDC3C7 !important; /* Lighter grey for smaller text */
}

</style>
'''

bot_template = '''
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.sideshow.com/storage/product-images/2171/c-3po_star-wars_square.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
<div class="chat-message user">
    <div class="avatar">
        <img src="https://resizing.flixster.com/ocuc8yjm8Fu5UK5Ze8lbdp58m9Y=/300x300/v2/https://flxt.tmsimg.com/assets/p11759522_i_h9_aa.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''