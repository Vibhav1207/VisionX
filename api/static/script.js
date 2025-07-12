const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const modal = document.getElementById('modal');
const outputImage = document.getElementById('outputImage');
const downloadLink = document.getElementById('downloadLink');
const closeBtn = document.getElementById('close');

fileInput.addEventListener('change', () => {
  fileName.textContent = fileInput.files[0]?.name || "Choose an image…";
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const operation = document.getElementById('operation').value;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('operation', operation);

  const res = await fetch('/process/', {
    method: 'POST',
    body: formData
  });

  const data = await res.json();
  const imgURL = `/static/output/${data.filename}`;

  outputImage.src = imgURL;
  downloadLink.href = imgURL;

  modal.style.display = 'block';
});

closeBtn.onclick = function () {
  modal.style.display = 'none';
};
window.onclick = function (event) {
  if (event.target == modal) {
    modal.style.display = 'none';
  }
};
