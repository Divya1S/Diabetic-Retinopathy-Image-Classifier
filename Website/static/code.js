$(document).ready(function () {
    $('form').on('submit', function (event) {
        event.preventDefault();
        var formData = new FormData($('form')[0]);
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                $('#result').text('Predicted class: ' + response.class + ', Confidence: ' + response.confidence.toFixed(2));
            }
        });
    });

    $('input[type="file"]').change(function (e) {
        var file = e.target.files[0];
        var reader = new FileReader();
        reader.onload = function (event) {
            $('#image-preview').attr('src', event.target.result);
            $('#image-preview').show(); // Show the image preview
        };
        reader.readAsDataURL(file);
    });
});
