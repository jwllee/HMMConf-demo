# Generated by Django 2.2.7 on 2019-11-15 14:50

from django.db import migrations, models
import hmmconfsite.demo.models
import sorl.thumbnail.fields


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0002_auto_20191115_1434'),
    ]

    operations = [
        migrations.AlterField(
            model_name='event',
            name='activity',
            field=models.CharField(max_length=500),
        ),
        migrations.AlterField(
            model_name='event',
            name='activity_id',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='event',
            name='file_barplot_logfwd',
            field=sorl.thumbnail.fields.ImageField(blank=True, max_length=500, upload_to=hmmconfsite.demo.models.get_file_barplot_logfwd_fp),
        ),
        migrations.AlterField(
            model_name='event',
            name='file_barplot_logfwd_before_obs',
            field=sorl.thumbnail.fields.ImageField(blank=True, max_length=500, upload_to=hmmconfsite.demo.models.get_file_barplot_logfwd_before_obs_fp),
        ),
        migrations.AlterField(
            model_name='event',
            name='file_net_logfwd',
            field=sorl.thumbnail.fields.ImageField(blank=True, max_length=500, upload_to=hmmconfsite.demo.models.get_file_net_logfwd_fp),
        ),
        migrations.AlterField(
            model_name='event',
            name='file_net_logfwd_before_obs',
            field=sorl.thumbnail.fields.ImageField(blank=True, max_length=500, upload_to=hmmconfsite.demo.models.get_file_net_logfwd_before_obs_fp),
        ),
        migrations.AlterField(
            model_name='log',
            name='file_confmat',
            field=models.FileField(max_length=500, upload_to=''),
        ),
        migrations.AlterField(
            model_name='log',
            name='file_logemitmat',
            field=models.FileField(max_length=500, upload_to=''),
        ),
        migrations.AlterField(
            model_name='log',
            name='file_logemitmat_d',
            field=models.FileField(max_length=500, upload_to=''),
        ),
        migrations.AlterField(
            model_name='log',
            name='file_logstartprob',
            field=models.FileField(max_length=500, upload_to=''),
        ),
        migrations.AlterField(
            model_name='log',
            name='file_logtranscube',
            field=models.FileField(max_length=500, upload_to=''),
        ),
        migrations.AlterField(
            model_name='log',
            name='file_logtranscube_d',
            field=models.FileField(max_length=500, upload_to=''),
        ),
        migrations.AlterField(
            model_name='log',
            name='file_pnml',
            field=models.FileField(max_length=500, upload_to=''),
        ),
        migrations.AlterField(
            model_name='log',
            name='name',
            field=models.CharField(max_length=500),
        ),
        migrations.AlterField(
            model_name='state',
            name='state',
            field=models.CharField(max_length=500),
        ),
    ]
