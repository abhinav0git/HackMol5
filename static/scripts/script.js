//new script

document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('file');
    const imagePreview = document.getElementById('image');
    const imageContainer = document.getElementById('something');  //after image jahan par show hogi

    // drag and drop events
    const dragArea = document.querySelector('.uploadsection');

    dragArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        dragArea.classList.add('drag-over');
    });

    dragArea.addEventListener('dragleave', function () {
        dragArea.classList.remove('drag-over');
    });

    dragArea.addEventListener('drop', function (e) {
        e.preventDefault();
        dragArea.classList.remove('drag-over');

        const droppedFiles = e.dataTransfer.files;
        fileInput.files = droppedFiles;

        handleFileSelect();
    });

    fileInput.addEventListener('change', handleFileSelect);

    function handleFileSelect() {
        const file = fileInput.files[0];
        const reader = new FileReader();    // to read the contents of the selected image file (specified by fileInput.files[0]) asynchronously

        reader.onload = function () {
            imagePreview.src = reader.result;
            imageContainer.style.display = 'block';
            svgImage.style.display = 'none'; // Hide the SVG image
        };

        reader.readAsDataURL(file);
    }
});
