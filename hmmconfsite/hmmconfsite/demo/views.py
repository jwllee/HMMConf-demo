from django.shortcuts import render
from django.views.generic.edit import FormView
from django.http import JsonResponse

from . import forms, models


def index(request):
    log = models.Log.objects.first()

    # there is no data, so make the user upload something
    if log is None:
        return UploadDataView.as_view()

    context = {
        'log_id': log.id
    }

    return render(request, 'demo/index.html', context)


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
