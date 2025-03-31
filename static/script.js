document.getElementById("uploadDoc").addEventListener("click", function () {
    document.getElementById("fileInput").click();
});

document.getElementById("fileInput").addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        uploadFile(file);
    }
});

function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    fetch('/upload', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("output").innerText = data.summary || "No text extracted.";
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById("output").innerText = "Error processing file.";
    });
}

document.getElementById("summarizeBtn").addEventListener("click", function () {
    const text = document.getElementById("textInput").value;

    fetch('/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("output").innerText = data.summary;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById("output").innerText = "Error summarizing text.";
    });
});

document.getElementById("sampleText").addEventListener("click", function () {
    document.getElementById("textInput").value = "Artificial intelligence (AI) is the simulation of human intelligence in machines. AI-powered applications are transforming various industries, making automation more efficient and effective.";
});


document.getElementById("pasteText").addEventListener("click", async function () {
    try {
        const text = await navigator.clipboard.readText();
        document.getElementById("textInput").value = text;
    } catch (err) {
        console.error("Failed to read clipboard contents:", err);
        alert("Clipboard access denied. Please paste manually.");
    }
});
