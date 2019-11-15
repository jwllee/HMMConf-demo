from django import forms



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


