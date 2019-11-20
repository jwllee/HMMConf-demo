/** 
 * Script for replaying events
 **/
function retrieve_record(event_id) {
    var url = $("div #retrieve-record").attr('data-retrieve-record-url');

    info_msg = 'Retrieving record for event: ' + event_id + ', url: ' + url;
    console.log(info_msg);

    $.ajax({
        url: url,
        data: {
            'csrfmiddlewaretoken': Cookies.get('csrftoken'),
            'event_id': event_id,
        },
        dataType: 'json',
        success: function(data) {
            info_msg = 'Retrieved record for event id: ' + data.event_id;
            console.log(info_msg);

            // Update figures
            $("#barplot-start-state").attr('src', data.barplot_state_url);
            $("#net-start-state").attr('src', data.net_state_url);

            // avoid simultaneously creation
            state_transition(data.event_id);
        }
    });
}

function state_transition(event_id) {
    var url = $("div #state-transition").attr('data-state-transition-url');

    info_msg = 'Getting images of state transition for event: ' + event_id + ', url: ' + url;
    console.log(info_msg);

    $.ajax({
        url: url,
        data: {
            'csrfmiddlewaretoken': Cookies.get('csrftoken'),
            'event_id': event_id,
        },
        dataType: 'json',
        success: function(data) {
            info_msg = 'Got images of state transition for event: ' + data.event_id;
            console.log(info_msg);

            // update figures
            $("#barplot-transition-state").attr('src', data.barplot_state_url);
            $("#net-transition-state").attr('src', data.net_state_url);

            // avoid simultaneously creation
            observation_update(data.event_id);
        }
    });
}

function observation_update(event_id) {
    var url = $("div #observation-update").attr('data-observation-update-url');

    info_msg = 'Getting images of observation update for event: ' + event_id + ', url: ' + url;
    console.log(info_msg);

    $.ajax({
        url: url,
        data: {
            'csrfmiddlewaretoken': Cookies.get('csrftoken'),
            'event_id': event_id,
        },
        dataType: 'json',
        success: function(data) {
            info_msg = 'Got images of observation update for event: ' + data.event_id;
            console.log(info_msg);

            // update figures
            $("#barplot-observation-state").attr('src', data.barplot_state_url);
            $("#net-observation-state").attr('src', data.net_state_url);

            // avoid simultaneous creation
            compute_conformance(data.event_id);
        }
    });
}

function compute_conformance(event_id) {
    var url = $("div #compute-conformance").attr('data-compute-conformance-url');

    info_msg = 'Getting images and table data of conformance computation for event: ' + event_id + ', url: ' + url;
    console.log(info_msg);

    $.ajax({
        url: url,
        data: {
            'csrfmiddlewaretoken': Cookies.get('csrftoken'),
            'event_id': event_id,
        },
        dataType: 'json',
        success: function(data) {
            info_msg = 'Got images and table data of conformance computation for event: ' + data.event_id;
            console.log(info_msg);

            // update figures
            $("#barplot-final-state").attr('src', data.barplot_state_url);
            $('#table-conformance').bootstrapTable('load', data.table_data);
        }
    });
}

$(document).ready(function() {
    var event_id = $("div #replay_detail").attr('data-event-id');
    retrieve_record(event_id);
    state_transition(event_id);
    observation_update(event_id);
    compute_conformance(event_id);
});

function replay_previous_event() {
    var event_id = $("div #replay_detail").attr('data-event-id');
    var url = $("div #replay_detail").attr('data-replay-previous-event-url');

    info_msg = 'Calling url: ' + url + ' to replay previous event before event: ' + event_id;
    console.log(info_msg);

    $.ajax({
        url: url,
        data: {
            'csrfmiddlewaretoken': Cookies.get('csrftoken'),
            'event_id': event_id,
        },
        dataType: 'json',
        success: function(data) {
            info_msg = 'Got previous event id: ' + data.event_id;
            console.log(info_msg);

            // update replay_detail
            var replay_detail = $("#replay_detail");
            replay_detail.attr("data-event-id", data.event_id);

            // update case event table
            $('#table-event').bootstrapTable('load', data.case_events);

            // update replay detail children here
            retrieve_record(data.event_id);
            state_transition(data.event_id);
            observation_update(data.event_id);
            compute_conformance(data.event_id);
        }
    });
}

function replay_next_event() {
    var event_id = $("div #replay_detail").attr('data-event-id');
    var url = $("div #replay_detail").attr('data-replay-next-event-url');

    info_msg = 'Calling url: ' + url + ' to replay next event after event: ' + event_id;
    console.log(info_msg);

    $.ajax({
        url: url,
        data: {
            'csrfmiddlewaretoken': Cookies.get('csrftoken'),
            'event_id': event_id,
        },
        dataType: 'json',
        success: function(data) {
            info_msg = 'Got next event id: ' + data.event_id;
            console.log(info_msg);

            // update replay_detail
            var replay_detail = $("#replay_detail");
            replay_detail.attr("data-event-id", data.event_id);

            // update case event table
            $('#table-event').bootstrapTable('load', data.case_events);

            // update replay detail children here
            retrieve_record(data.event_id);
        }
    });
}
