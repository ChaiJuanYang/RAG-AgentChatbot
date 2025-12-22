async function sendMessage() {
    const input = document.getElementById("user-input");
    const chatbox = document.getElementById("chat-box");
    const message = input.value.trim();
    if (!message) return;

    appendMessage("user-message", message);
    input.value = "";
    addLoadingMessage();
    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
    });

    const data = await response.json();
    removeLoadingMessage();
    appendMessage("bot-message", data.answer);
}

function appendMessage(sender, text) {
    const chatBox = document.getElementById("chat-box");
    const msg = document.createElement("div");
    msg.innerHTML = `<b>${sender}:</b> ${text}`;

    msg.classList.add("message", sender);
    msg.innerText = text;
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addLoadingMessage() {
  const chatBox = document.getElementById("chat-box");

  const loadingDiv = document.createElement("div");
  loadingDiv.classList.add("message", "bot-message");
  loadingDiv.id = "loading-message";

  loadingDiv.innerHTML = `
    <div class="loading">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </div>
  `;

  chatBox.appendChild(loadingDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}
function removeLoadingMessage() {
  const loading = document.getElementById("loading-message");
  if (loading) loading.remove();
}
