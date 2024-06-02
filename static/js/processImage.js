document.getElementById('processForm').addEventListener('submit', function(event) {
    event.preventDefault();
    Swal.fire({
        title: 'Processing...',
        html: 'The image is being processed. Please wait',
        timerProgressBar: true,
        allowOutsideClick: false,
        allowEscapeKey: false,
        showConfirmButton: false,
        willOpen: () => {
            Swal.showLoading();
        },
    });
    this.submit();
});