# Generated by Django 2.2.7 on 2019-11-20 03:31

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0006_auto_20191119_1341'),
    ]

    operations = [
        migrations.AlterField(
            model_name='event',
            name='mode_state_before_obs',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='mode_state_before_obs', to='demo.State'),
        ),
    ]
