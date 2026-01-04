<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
    <title>Secure Llama</title>
    <style>
        /* [Your existing CSS styles here] */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            height: 100%;
            width: 100%;
            overflow: hidden;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #121212;
            color: #e0e0e0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            width: 95%;
            max-width: 600px;
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 8px 0;
        }

        .header {
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
        }

        .title {
            font-size: 20px;
            font-weight: bold;
            color: #4caf50;
        }

        .about-link {
            padding: 6px 10px;
            background: #333;
            color: #e0e0e0;
            border: 1px solid #444;
            border-radius: 6px;
            cursor: pointer;
            text-decoration: none;
            font-size: 14px;
        }

        .slider-container {
            display: flex;
            flex-direction: column;
            gap: 8px;
            background: #1e1e1e;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #444;
        }

        .slider-label {
            font-size: 13px;
        }

        .slider-controls {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .slider {
            flex: 1;
            -webkit-appearance: none;
            height: 4px;
            background: #444;
            border-radius: 2px;
            outline: none;
        }

        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #4caf50;
            cursor: pointer;
        }

        .slider-value {
            font-size: 13px;
            min-width: 30px;
        }

        #chat-container {
            flex: 1;
            background: #1e1e1e;
            border-radius: 8px;
            border: 1px solid #333;
            overflow-y: auto;
            padding: 10px;
            margin: 4px 0;
        }

        .message {
            padding: 8px 10px;
            border-radius: 8px;
            margin: 4px 0;
            max-width: 85%;
            word-wrap: break-word;
            font-size: 14px;
        }

        .user-message {
            align-self: flex-end;
            background: #4caf50;
            color: #fff;
            margin-left: auto;
        }

        .bot-message {
            align-self: flex-start;
            background: #333;
            color: #e0e0e0;
        }

        #input-container {
            display: flex;
            gap: 4px;
            width: 100%;
            padding: 4px 0;
        }

        #message-input {
            flex: 1;
            padding: 8px 10px;
            font-size: 14px;
            background: #1e1e1e;
            color: #e0e0e0;
            border: 1px solid #333;
            border-radius: 6px;
            outline: none;
        }

        #send-button {
            padding: 8px 12px;
            background: #444;
            color: #fff;
            font-size: 14px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }

        @media (min-width: 768px) {
            .container {
                gap: 10px;
                padding: 10px 0;
            }

            .slider-container {
                flex-direction: row;
                align-items: center;
                padding: 12px 15px;
            }

            .slider-label {
                font-size: 14px;
                white-space: nowrap;
                min-width: 200px;
            }

            .message {
                padding: 10px 12px;
                font-size: 15px;
            }

            #message-input, #send-button {
                padding: 10px 12px;
                font-size: 15px;
            }

            .title {
                font-size: 24px;
            }

            .about-link {
                padding: 8px 15px;
                font-size: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">Secure Llama</div>
            <a href="about.html" class="about-link">About</a>
        </div>
        <!-- -->
        <div style="color: #ff4444; font-size: 0.7em; display: block;">
            API is currently disabled to avoid incurring costs on AWS
        </div>

        <div class="slider-container">
            <label class="slider-label">Embedding L2-Norm Noise Ratio:</label>
            <div class="slider-controls">
                <input type="range" min="0" max="1" step="0.01" value="0.5" class="slider" id="noise-ratio">
                <span class="slider-value" id="noise-ratio-value">0.5</span>
            </div>
        </div>

        <div id="chat-container"></div>
        
        <div id="input-container">
            <input type="text" id="message-input" placeholder="Type your message here..." />
            <button id="send-button">Send</button>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById("chat-container");
        const messageInput = document.getElementById("message-input");
        const sendButton = document.getElementById("send-button");
        const noiseRatioSlider = document.getElementById("noise-ratio");
        const noiseRatioValue = document.getElementById("noise-ratio-value");

        noiseRatioSlider.addEventListener("input", (e) => {
            noiseRatioValue.textContent = parseFloat(e.target.value).toFixed(2);
        });

        // Placeholders for server URL and credentials
        const serverStreamUrl = "${server_url}"; 
        const username = "${username}";
        const password = "${password}";

        let chatHistory = [];

        /**
         * Append a message to the chat UI.
         * @param {string} content - Text to display
         * @param {boolean} isUser - If true, style as user message; otherwise bot
         */
        function appendMessage(content, isUser = true) {
            const message = document.createElement("div");
            message.className = "message " + (isUser ? "user-message" : "bot-message");
            message.textContent = content;
            chatContainer.appendChild(message);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            // Update chat history
            chatHistory.push({ content, isUser });

            // Save updated chat history to sessionStorage
            sessionStorage.setItem("chatHistory", JSON.stringify(chatHistory));
        }

        /**
         * Construct the history string in the format:
         * "Human: {chat1 request} Assistant: {chat1 response} Human: {chat2 request} Assistant: {chat2 response} ..."
         * @returns {string} - Formatted history string
         */
        function constructHistory() {
            return chatHistory.map(msg => {
                return msg.isUser ? "Human: " + msg.content : "Assistant: " + msg.content;
            }).join(' ');
        }

        /**
         * Send the user's prompt to the server (streaming),
         * and append tokens as they arrive.
         */
        async function sendMessage() {
            const userMessage = messageInput.value.trim();
            if (!userMessage) return;

            // 1. Display user's message in UI
            appendMessage(userMessage, true);
            messageInput.value = "";

            // 2. Prepare a placeholder for the streaming bot response
            const botMessageEl = document.createElement("div");
            botMessageEl.className = "message bot-message";
            botMessageEl.textContent = "...";
            chatContainer.appendChild(botMessageEl);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                // 3. Construct the history string
                const historyString = constructHistory();

                // 4. Set up the POST body for your streaming endpoint
                const requestData = {
                    prompt: userMessage,
                    max_length: 512,
                    temperature: 1.0,
                    l2_norm: parseFloat(noiseRatioSlider.value),
                    history: historyString
                };

                // 5. Make the fetch call to /generate_stream
                const response = await fetch(serverStreamUrl, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": "Basic " + btoa(username + ":" + password)
                    },
                    body: JSON.stringify(requestData)
                });

                // Check if the response is OK
                if (!response.ok) {
                    const errorText = await response.text();
                    botMessageEl.textContent =
                        "Error: Server responded with status " +
                        response.status + " - " + errorText;
                    return;
                }

                // 6. Read the response as a stream of text
                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let doneReading = false;

                // We'll accumulate chunks here
                let partialText = "";

                while (!doneReading) {
                    const { value, done } = await reader.read();
                    if (done) {
                        doneReading = true;
                        break;
                    }
                    // Convert Uint8Array chunk to string with replacement for invalid chars
                    const chunk = decoder.decode(value, { stream: true });
                    partialText += chunk;

                    // Check if "Human:" is in the latest part of partialText
                    const assistantResponse = partialText.split("Assistant:").pop() || partialText;
                    if (assistantResponse.includes("Human:")) {
                        // Remove everything after "Human:"
                        partialText = partialText.split("Human:")[0];
                        doneReading = true;
                    }

                    // Update the bot message element with the partial text
                    botMessageEl.textContent = partialText;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }

                // 7. Append the final AI response to chatHistory
                appendMessage(partialText, false);
                // Remove the placeholder
                chatContainer.removeChild(botMessageEl);

            } catch (error) {
                botMessageEl.textContent =
                    "Error: Network error or server is unreachable. Details: " + error.message;
            }
        }

        sendButton.addEventListener("click", sendMessage);
        messageInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                sendMessage();
            }
        });

        // **Added Code Starts Here**

        /**
         * Load chat history from sessionStorage on page load
         */
        window.addEventListener("DOMContentLoaded", () => {
            const storedChatHistory = sessionStorage.getItem("chatHistory");
            if (storedChatHistory) {
                try {
                    chatHistory = JSON.parse(storedChatHistory);
                    chatHistory.forEach(msg => appendMessage(msg.content, msg.isUser));
                } catch (e) {
                    console.error("Failed to parse chat history from sessionStorage:", e);
                    sessionStorage.removeItem("chatHistory");
                    chatHistory = [];
                }
            }
        });

        /**
         * Clear chat history from sessionStorage when the page is refreshed or navigated away
         */
        window.addEventListener("beforeunload", () => {
            sessionStorage.removeItem("chatHistory");
        });

        // **Added Code Ends Here**
    </script>

</body>
</html>
