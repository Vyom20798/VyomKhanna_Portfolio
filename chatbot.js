const chatButton = document.getElementById('chat-button');
const chatPopup = document.getElementById('chat-popup');
const closeChat = document.getElementById('close-chat');
const sendButton = document.getElementById('send-button');
const userInput = document.getElementById('user-input');
const chatMessages = document.getElementById('chat-messages');

// Open Chat
chatButton.addEventListener('click', () => {
  chatPopup.style.display = 'flex';
  addMessage('bot', 'Namaste! 👋 How can I help you explore my portfolio?');
});

// Close Chat
closeChat.addEventListener('click', () => {
  chatPopup.style.display = 'none';
  chatMessages.innerHTML = '';
});

// Send Message
sendButton.addEventListener('click', async () => {
  const message = userInput.value.trim();
  if (message) {
    addMessage('user', message);
    userInput.value = '';

    // Call backend API
    const response = await fetch('https://chatbot-theta-dun.vercel.app/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: message })
    });     

    const data = await response.json();
    addMessage('bot', data.reply);
  }
});

// Add Message to Chat
function addMessage(sender, text) {
  const msg = document.createElement('div');
  msg.className = sender === 'user' ? 'user-message' : 'bot-message';
  msg.innerText = text;
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
