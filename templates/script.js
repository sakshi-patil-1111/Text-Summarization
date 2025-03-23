document.getElementById("sampleText").addEventListener("click", function () {
    document.getElementById("textInput").value = "This is a sample text for AI summarization.";
});

document.getElementById("pasteText").addEventListener("click", async function () {
    try {
        const text = await navigator.clipboard.readText();
        document.getElementById("textInput").value = text;
    } catch (err) {
        alert("Failed to paste text. Please allow clipboard permissions.");
    }
});

document.getElementById("uploadDoc").addEventListener("click", function () {
    document.getElementById("fileInput").click(); // Opens file dialog
});

