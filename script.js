function uploadImage() {
    let fileInput = document.getElementById("fileInput");
    let file = fileInput.files[0];
    
    if (!file) {
        alert("Please select an image file.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("result").innerText = "Error: " + data.error;
        } else {
            document.getElementById("result").innerText = `Result: ${data.result} (Confidence: ${data.confidence}%)`;
        }
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Prediction failed.";
    });
}
