function confirmDiscardImages(endpoint) {
    let discardImageConfiguration = {
        title: 'Are you sure?',
        icon: 'warning',
        allowOutsideClick: false,
        allowEscapeKey: false,
        showCancelButton: true,
        confirmButtonColor: '#0000e6',
        cancelButtonColor: '#e60000',
        confirmButtonText: 'Yes',
        cancelButtonText: 'No'
    };
    let errorConfiguration={
        title : 'Error',
        icon: 'error',
        allowOutsideClick: false,
        allowEscapeKey: false,
        showCancelButton: true,
        confirmButtonColor: '#0000e6',
    };
    Swal.fire(discardImageConfiguration).then((result) => {
        if (result.isConfirmed) {
            fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            })
            .then(response => {
                if (response.redirected) {
                    window.location.href = response.url;
                } else {
                    return response.json();
                }
            })
            .catch(error => {
                errorConfiguration.text = error;
                Swal.fire(errorConfiguration);
            });
        }
    });
}

function submitSaveImageForm() {
    window.onload = function() {
        document.cookie = 'fileDownload=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    };

    document.getElementById('saveImageForm').submit();

    var checkCookie = setInterval(function() {
        if (document.cookie.split(';').some((item) => item.trim().startsWith('fileDownload='))) {
            window.location.href = '/';
            document.cookie = 'fileDownload=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
            clearInterval(checkCookie);
        }
    }, 500);
}

function showSlides(n)
{
    let processedImages = document.getElementsByClassName("processedImages");
    if (n > processedImages.length)
    {
        imageIndex = 1
    }
    if (n < 1)
    {
        imageIndex = processedImages.length
    }
    for (let i = 0; i < processedImages.length; i++)
    {
    processedImages[i].style.display = "none";
    }
    processedImages[imageIndex-1].style.display = "block";
}

function previousNextImage(n)
{
    showSlides(imageIndex += n);
}

let imageIndex = 1;
showSlides(imageIndex);
