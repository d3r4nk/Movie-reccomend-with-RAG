const form = document.querySelector("#query-form");
const queryInput = document.querySelector("#query");
const topKInput = document.querySelector("#top-k");
const topKValue = document.querySelector("#top-k-value");
const submitButton = document.querySelector("#submit-button");
const answerBox = document.querySelector("#answer");
const resultsBox = document.querySelector("#results");
const resultCount = document.querySelector("#result-count");
const statusPill = document.querySelector("#status-pill");

function setStatus(text, mode = "ready") {
  statusPill.textContent = text;
  statusPill.dataset.mode = mode;
}

function renderResults(results) {
  resultCount.textContent = results.length;
  resultsBox.innerHTML = "";

  if (!results.length) {
    resultsBox.innerHTML = '<p class="empty-state">No chunks retrieved yet.</p>';
    return;
  }

  for (const item of results) {
    const article = document.createElement("article");
    article.className = "result-item";

    const distance = Number(item.Distance).toFixed(4);
    article.innerHTML = `
      <div class="result-header">
        <h3></h3>
        <span>${distance}</span>
      </div>
      <p></p>
    `;
    article.querySelector("h3").textContent = item.Title || "Untitled";
    article.querySelector("p").textContent = item.Chunk || "";
    resultsBox.appendChild(article);
  }
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderAnswer(markdownText) {
  const escaped = escapeHtml(markdownText || "");
  answerBox.innerHTML = escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }

  submitButton.disabled = true;
  setStatus("Thinking", "loading");
  renderAnswer("Retrieving similar movie chunks and asking the local model...");
  renderResults([]);

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: Number(topKInput.value),
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.details || data.error || "Request failed");
    }

    renderAnswer(data.answer);
    renderResults(data.results || []);
    setStatus("Ready", "ready");
  } catch (error) {
    renderAnswer(error.message);
    setStatus("Error", "error");
  } finally {
    submitButton.disabled = false;
  }
});
