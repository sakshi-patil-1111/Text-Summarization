document.getElementById("uploadDoc").addEventListener("click", () => {
    document.getElementById("fileInput").click();
});

document.getElementById("fileInput").addEventListener("change", function () {
    const file = this.files[0];
    if (file) uploadFile(file, "/upload");
});

document.getElementById("summarizeBtn").addEventListener("click", () => {
    const text = document.getElementById("textInput").value;

    fetch("/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("output").innerText = data.summary || "No summary generated.";
    })
    .catch(err => {
        console.error("Summarization error:", err);
        document.getElementById("output").innerText = "Error summarizing text.";
    });
});

document.getElementById("sampleText").addEventListener("click", () => {
    document.getElementById("textInput").value =
        "Artificial intelligence (AI) is the simulation of human intelligence in machines. AI-powered applications are transforming various industries, making automation more efficient and effective.";
});

document.getElementById("pasteText").addEventListener("click", async () => {
    try {
        const text = await navigator.clipboard.readText();
        document.getElementById("textInput").value = text;
    } catch (err) {
        console.error("Clipboard error:", err);
        alert("Clipboard access denied. Please paste manually.");
    }
});

document.getElementById("evaluateBtn").addEventListener("click", () => {
    document.getElementById("csvInput").click();
});

document.getElementById("csvInput").addEventListener("change", function () {
    const file = this.files[0];
    if (file) uploadFile(file, "/evaluate-summary");
});

function uploadFile(file, endpoint) {
    const formData = new FormData();
    formData.append("file", file);

    fetch(endpoint, {
        method: "POST",
        body: formData,
    })
    .then(res => res.json())
    .then(data => {
        if (endpoint === "/upload") {
            document.getElementById("output").innerText = data.summary || "No summary generated.";
        } else if (endpoint === "/evaluate-summary") {
            if (data.scores) {
                const output = Object.entries(data.scores)
                    .map(([key, value]) => `${key.toUpperCase()}: ${value}`)
                    .join("\n");
                document.getElementById("evaluationOutput").innerText = output;
            }
        }
    })
    .catch(err => {
        console.error("Upload error:", err);
        const target = endpoint === "/evaluate-summary" ? "evaluationOutput" : "output";
        document.getElementById(target).innerText = "Error processing file.";
    });
}
