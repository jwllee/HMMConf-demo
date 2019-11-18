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


def replay_next_event(request):
    event_id = request.GET.get('event_id', -1)
    event = models.Event.objects.get(pk=event_id)
    assert isinstance(event, models.Event)

    caseid = event.caseid
    next_event = models.Event.objects.filter(
        caseid__exact=caseid,
        index__gt=event.index,
    ).order_by('index').first()

    # if there is no next event, just use the current event
    if not next_event:
        next_event = event

    data = {
        'event_id': next_event.id,
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
            'index': event_i.index,
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
