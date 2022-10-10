function readURL(input) {

    if (input.files && input.files[0]) {

        var reader = new FileReader();
      
        reader.onload = function (e) {
            $('#your-image')
                .attr('src', e.target.result)
                .width(700)
                .height(500);

        };
        reader.readAsDataURL(input.files[0]);
    }
    }
    