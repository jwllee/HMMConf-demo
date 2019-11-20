from django import forms


from . import models


def validate_filename_extension(filename, *exts):
    cond = False
    for ext in exts:
        cond = cond or filename.endswith(ext)
    if not cond:
        s = ' or '.join(exts)
        err_msg = 'File extension should be {}'.format(s)
        raise forms.ValidationError(err_msg)


def validate_file_pnml(file_):
    validate_filename_extension(file_.name, '.pnml', '.apnml')


def validate_file_npy(file_):
    validate_filename_extension(file_.name, '.npy')


def validate_file_csv(file_):
    validate_filename_extension(file_.name, '.csv')


class UploadDataForm(forms.Form):
    file_pnml = forms.FileField(validators=[validate_file_pnml])
    file_logstartprob = forms.FileField(validators=[validate_file_npy])
    file_logtranscube = forms.FileField(validators=[validate_file_npy])
    file_logtranscube_d = forms.FileField(validators=[validate_file_npy])
    file_logemitmat = forms.FileField(validators=[validate_file_npy])
    file_logemitmat_d = forms.FileField(validators=[validate_file_npy])
    file_confmat = forms.FileField(validators=[validate_file_npy])
    file_state_csv = forms.FileField(validators=[validate_file_csv])
    file_event_csv = forms.FileField(validators=[validate_file_csv])


class CaseChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj):
        str_ = '{:<8} - {}'
        is_train = 'train' if obj.is_train else 'test'
        str_ = str_.format(obj.caseid, is_train)
        return str_


class CaseChoiceSelectForm(forms.Form):
    case_select = CaseChoiceField(queryset=None, label='Replay case')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        first_events = models.Event.objects.filter(index__exact=0).order_by('id')
        self.fields['case_select'].queryset = first_events
