document.addEventListener('DOMContentLoaded', function() {
    const alerts = document.querySelectorAll('.alert-messages');
    alerts.forEach(alert => {
        const message = alert.dataset.message;
        const category = alert.dataset.category;

        let defaultConfiguration = {
            text: message,
            allowOutsideClick: false,
            allowEscapeKey: false
        };

        switch (category){
            case 'success':
                defaultConfiguration.title='Success';
                defaultConfiguration.icon='success';
                break;
            case 'error':
                defaultConfiguration.title='Error';
                defaultConfiguration.icon='error';
                break;
            case 'info':
                defaultConfiguration.title='Information';
                defaultConfiguration.icon='info';
                break;
            case 'imageSaved':
                defaultConfiguration.title='Success';
                defaultConfiguration.icon='success';
                defaultConfiguration.showConfirmButton=false;
                defaultConfiguration.timer=1500;
                defaultConfiguration.position='center';
                break;
        }

        Swal.fire(defaultConfiguration);

    });
});