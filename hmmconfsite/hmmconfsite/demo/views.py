from django.shortcuts import render
from django.views.generic.edit import FormView

from . import forms, models


def index(request):
    context = {

    }
    return render(request, 'demo/index.html', context)


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
