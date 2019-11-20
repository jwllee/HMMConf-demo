from django.shortcuts import render
from django.views.generic.edit import FormView
from django.http import JsonResponse, Http404

from . import forms, models
from .utils import make_logger


logger = make_logger('views.py')


def index(request):
    log = models.Log.objects.first()

    # there is no data, so make the user upload something
    if log is None:
        return UploadDataView.as_view()

    if request.method == 'POST':
        case_select_form = forms.CaseChoiceSelectForm(request.POST)

        if case_select_form.is_valid():
            event = case_select_form.cleaned_data['case_select']
            event_id = event.id
        else:
            raise Http404('Case ID not found.')
    else:
        # try to get caseid and event
        caseid = request.GET.get('caseid', None)

        if caseid is None:
            event = models.Event.objects.filter(index__exact=0).first()
            event_id = event.id
        else:
            event_id = request.GET.get('event_id')
            event = models.Event.objects.filter(
                caseid__exact=event.caseid,
                index__exact=0,
            ).first()

        initial = {
            'case_select': event
        }
        case_select_form = forms.CaseChoiceSelectForm(initial=initial)

    context = {
        'log_id': log.id,
        'case_select_form': case_select_form,
        'caseid': event.caseid,
        'event_id': event_id,
    }

    return render(request, 'demo/index.html', context)


def about(request):
    context = {

    }
    return render(request, 'demo/about.html', context)


def replay_previous_event(request):
    event_id = request.GET.get('event_id', -1)
    event = models.Event.objects.get(pk=event_id)
    assert isinstance(event, models.Event)

    caseid = event.caseid
    prev_event = models.Event.objects.filter(
        caseid__exact=caseid,
        index__exact=event.index - 1,
    ).order_by('index').first()

    # if there is no next event, just use the current event
    if not prev_event:
        prev_event = event

    # need to update case event table data
    case_events = models.Event.objects.filter(
        caseid__exact=caseid
    ).order_by('index')

    rows = []
    for event_i in case_events:
        row = {
            # so that it's not 0-index
            'index': event_i.index + 1,
            'activity_label': event_i.activity,
            'current': event_i.id == prev_event.id,
        }
        rows.append(row)

    data = {
        'event_id': prev_event.id,
        'case_events': rows,
    }

    return JsonResponse(data)


def replay_next_event(request):
    event_id = request.GET.get('event_id', -1)
    event = models.Event.objects.get(pk=event_id)
    assert isinstance(event, models.Event)

    caseid = event.caseid
    next_event = models.Event.objects.filter(
        caseid__exact=caseid,
        index__exact=event.index + 1,
    ).order_by('index').first()

    # if there is no next event, just use the current event
    if not next_event:
        next_event = event

    # need to update case event table data
    case_events = models.Event.objects.filter(
        caseid__exact=caseid
    ).order_by('index')

    rows = []
    for event_i in case_events:
        row = {
            # so that it's not 0-index
            'index': event_i.index + 1,
            'activity_label': event_i.activity,
            'current': event_i.id == next_event.id,
        }
        rows.append(row)

    data = {
        'event_id': next_event.id,
        'case_events': rows,
    }

    return JsonResponse(data)


def json_event_stream_data(request, event_id):
    event = models.Event.objects.get(pk=event_id)
    caseid = event.caseid
    case_events = models.Event.objects.filter(
        caseid__exact=caseid
    ).order_by('index')

    rows = []
    for event_i in case_events:
        row = {
            # so that it's not 0-index
            'index': event_i.index + 1,
            'activity_label': event_i.activity,
            'current': event_i.id == event.id
        }
        rows.append(row)

    data = {
        'rows': rows
    }

    return JsonResponse(data)


def get_barplot_case(request):
    log_id = request.GET.get('log_id', -1)

    log = models.Log.objects.get(pk=log_id)
    assert isinstance(log, models.Log)

    case_length_file = log.get_barplot_case_length()
    unique_activity_file = log.get_barplot_case_unique_activity()

    data = {
        'barplot_case_length_url': case_length_file.url,
        'barplot_case_length_name': case_length_file.name,
        'barplot_case_unique_activity_url': unique_activity_file.url,
        'barplot_case_unique_activity_name': unique_activity_file.name,
    }
    return JsonResponse(data)


class UploadDataView(FormView):
    template_name = 'demo/upload_data.html'
    form_class = forms.UploadDataForm
    success_url = '/'

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        assert isinstance(form, forms.UploadDataForm)

        file_pnml = form.cleaned_data['file_pnml']
        file_logstartprob = form.cleaned_data['file_logstartprob']
        file_logtranscube = form.cleaned_data['file_logtranscube']
        file_logtranscube_d = form.cleaned_data['file_logtranscube_d']
        file_logemitmat = form.cleaned_data['file_logemitmat']
        file_logemitmat_d = form.cleaned_data['file_logemitmat_d']
        file_confmat = form.cleaned_data['file_confmat']
        file_state_csv = form.cleaned_data['file_state_csv']
        file_event_csv = form.cleaned_data['file_event_csv']

        models.create_objects(
            file_pnml, file_logstartprob,
            file_logtranscube, file_logtranscube_d,
            file_logemitmat, file_logemitmat_d,
            file_confmat, file_state_csv,
            file_event_csv, row_limit=10000
        )

        return super().form_valid(form)


def retrieve_record(request):
    event_id = request.GET.get('event_id', -1)
    event = models.Event.objects.get(pk=event_id)
    assert isinstance(event, models.Event)

    # get previous state estimation or initial estimation
    if event.index == 0:
        # get initial estimation
        file_barplot_state = event.log.get_file_barplot_logstartprob()
        file_net_state = event.log.get_file_net_logstartprob()
    else:
        # the previous event's final state estimation
        prev_event = models.Event.objects.filter(
            index__exact=event.index - 1,
            caseid__exact=event.caseid
        ).first()
        assert isinstance(prev_event, models.Event)
        file_barplot_state = prev_event.get_file_barplot_logfwd()
        file_net_state = prev_event.get_file_net_logfwd()

    data = {
        'event_id': event_id,
        'barplot_state_url': file_barplot_state.url,
        'barplot_state_name': file_barplot_state.name,
        'net_state_url': file_net_state.url,
        'net_state_name': file_net_state.name,
    }

    return JsonResponse(data)


def state_transition(request):
    event_id = request.GET.get('event_id', -1)
    event = models.Event.objects.get(pk=event_id)
    assert isinstance(event, models.Event)

    # just return the current estimation if it's the first event
    # otherwise do the state transition
    if event.index == 0:
        return retrieve_record(request)
    else:
        file_barplot_state = event.get_file_barplot_logfwd_before_obs()
        file_net_state = event.get_file_net_logfwd_before_obs()

        data = {
            'event_id': event_id,
            'barplot_state_url': file_barplot_state.url,
            'barplot_state_name': file_barplot_state.name,
            'net_state_url': file_net_state.url,
            'net_state_name': file_net_state.name,
        }

        return JsonResponse(data)


def observation_update(request):
    event_id = request.GET.get('event_id', -1)
    event = models.Event.objects.get(pk=event_id)
    assert isinstance(event, models.Event)

    file_barplot_state = event.get_file_barplot_logfwd()
    file_net_state = event.get_file_net_logfwd()

    data = {
        'event_id': event_id,
        'barplot_state_url': file_barplot_state.url,
        'barplot_state_name': file_barplot_state.name,
        'net_state_url': file_net_state.url,
        'net_state_name': file_net_state.name,
    }

    return JsonResponse(data)


def get_conformance_data(event_id):
    event = models.Event.objects.get(pk=event_id)
    assert isinstance(event, models.Event)

    if event.index == 0:
        prev_event_str = '-'
    else:
        prev_event = models.Event.objects.filter(
            index__exact=event.index - 1,
            caseid__exact=event.caseid,
        ).first()
        prev_event_str = prev_event.activity

    case_events = models.Event.objects.filter(
        caseid__exact=event.caseid,
        log__exact=event.log,
        index__lte=event.index,
    )

    # compute total injected distance
    total_injected_distance = case_events.values_list('injected_distance', flat=True)
    total_injected_distance = sum(total_injected_distance)

    # compute avg. conformance
    avgconf = list(case_events.values_list('finalconf', flat=True))
    avgconf = sum(avgconf) / len(avgconf)

    # conformance table data
    row_caseid = {
        'attribute': 'Case ID',
        'value': event.caseid
    }
    row_prev_event = {
        'attribute': 'Previous event',
        'value': prev_event_str,
    }
    row_event = {
        'attribute': 'Current event',
        'value': event.activity,
    }
    row_finalconf = {
        'attribute': 'Final conformance',
        'value': '{:.5f}'.format(event.finalconf),
    }
    row_injected_distance = {
        'attribute': 'Injected distance',
        'value': '{:.5f}'.format(event.injected_distance),
    }
    row_total_injected_distance = {
        'attribute': 'Total injected distance',
        'value': '{:.5f}'.format(total_injected_distance),
    }
    row_completeness = {
        'attribute': 'Completeness',
        'value': '{:.5f}'.format(event.completeness),
    }
    row_avg_conformance = {
        'attribute': 'Avg. conformance',
        'value': '{:.5f}'.format(avgconf),
    }
    table_data = [
        row_caseid,
        row_prev_event,
        row_event,
        row_finalconf,
        row_injected_distance,
        row_total_injected_distance,
        row_completeness,
        row_avg_conformance
    ]

    return table_data


def json_conformance_data(request, event_id):
    table_data = get_conformance_data(event_id)

    data = {
        'table_data': table_data
    }

    return JsonResponse(data)


def compute_conformance(request):
    event_id = request.GET.get('event_id', -1)
    event = models.Event.objects.get(pk=event_id)
    assert isinstance(event, models.Event)

    file_barplot_state = event.get_file_barplot_logfwd()

    table_data = get_conformance_data(event_id)

    data = {
        'event_id': event_id,
        'barplot_state_url': file_barplot_state.url,
        'barplot_state_name': file_barplot_state.name,
        'table_data': table_data,
    }

    return JsonResponse(data)
