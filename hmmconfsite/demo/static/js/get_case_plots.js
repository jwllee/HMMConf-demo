// script to update index.html case distribution barplots
$(document).ready(function() {
    var get_barplot_url = $('div #barplot').attr('data-get-barplot-url');
    var log_id = $('div #barplot').attr('data-log-id');
    var data = {
        'csrfmiddlewaretoken': Cookies.get('csrftoken'),
        'log_id': log_id,
    };

    $.ajax({
        url: get_barplot_url,
        data: data,
        dataType: 'json',
        success: function(data) {
            var case_length_url = data.barplot_case_length_url;
            var case_length_name = data.barplot_case_length_name;
            var unique_activity_url = data.barplot_case_unique_activity_url;
            var unique_activity_name = data.barplot_case_unique_activity_name;

            info_msg = 'Bar plot success: case_length_url: ' + case_length_url;
            info_msg = info_msg + ' unique_activity_url: ' + unique_activity_url;
            console.log(info_msg);

            var case_length_image = '<img class="img-fluid" src="' + case_length_url;
            case_length_image = case_length_image + '" alt="Case length distribution">';
            $('div #barplot_case_length').append(case_length_image);

            var unique_activity_image = '<img class="img-fluid" src="' + unique_activity_url;
            unique_activity_image = unique_activity_image + '" alt="Unique activity distribution">';
            $('div #barplot_unique_activity').append(unique_activity_image);
        }
    });

});
