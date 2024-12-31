def get_html_template():
    return """
    <style>
        body {
            background-color: #000;
            color: #fff;
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #111;
            color: #fff;
        }
        .chat-bubble {
            display: flex;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            border-radius: 15px;
            max-width: 75%;
        }
        .user-bubble {
            background-color: #444;
            align-self: flex-end;
            color: #fff;
        }
        .bot-bubble {
            background-color: #222;
            align-self: flex-start;
            color: #ddd;
        }
        .chat-bubble span {
            margin-left: 10px;
        }
        .sidebar .sidebar-content h1,
        .sidebar .sidebar-content h2,
        .sidebar .sidebar-content h3 {
            color: #fff;
        }
    </style>
    """
