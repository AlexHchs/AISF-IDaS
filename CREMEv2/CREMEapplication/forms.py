from dataclasses import fields
from django import forms

from .models import Index, Testbed, Controller, DataLoggerServer, TargetServer, BenignServer, VulnerableClient,\
    NonVulnerableClient, AttackerServer, MaliciousClient, AttackScenario, MachineLearningModel, SkipStage


class IndexForm(forms.ModelForm):
    class Meta:
        model = Index
        fields = ('number_of_UE',)


class TestbedForm(forms.ModelForm):
    class Meta:
        model = Testbed
        fields = ('UE_id', 'number_of_controller', 'number_of_data_logger_server', 'number_of_target_server',
                  'number_of_benign_server', 'number_of_vulnerable_client', 'number_of_non_vulnerable_client',
                  'number_of_attacker_server', 'number_of_malicious_client')
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True
        self.fields['number_of_controller'].disabled = True
        self.fields['number_of_data_logger_server'].disabled = True
        self.fields['number_of_target_server'].disabled = True
        self.fields['number_of_benign_server'].disabled = True
        self.fields['number_of_vulnerable_client'].disabled = True
        self.fields['number_of_non_vulnerable_client'].disabled = True
        self.fields['number_of_attacker_server'].disabled = True
        self.fields['number_of_malicious_client'].disabled = True


class ControllerForm(forms.ModelForm):
    class Meta:
        model = Controller
        fields = ('UE_id', 'hostname', 'ip', 'username', 'password', 'path')
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True


class DataLoggerServerForm(forms.ModelForm):
    class Meta:
        model = DataLoggerServer
        fields = ('UE_id', 'hostname', 'ip', 'username', 'password', 'path', 'network_interface', 'atop_interval')
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True


class TargetServerForm(forms.ModelForm):
    class Meta:
        model = TargetServer
        fields = ('UE_id', 'hostname', 'ip', 'username', 'password', 'path')
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True


class BenignServerForm(forms.ModelForm):
    class Meta:
        model = BenignServer
        fields = ('UE_id', 'hostname', 'ip', 'username', 'password', 'path')
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True


class VulnerableClientForm(forms.ModelForm):
    class Meta:
        model = VulnerableClient
        fields = ('UE_id', 'hostname', 'ip', 'username', 'password', 'path')
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True


class NonVulnerableClientForm(forms.ModelForm):
    class Meta:
        model = NonVulnerableClient
        fields = ('UE_id', 'hostname', 'ip', 'username', 'password', 'path')
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True


class AttackerServerForm(forms.ModelForm):
    class Meta:
        model = AttackerServer
        fields = ('UE_id', 'hostname', 'ip', 'username', 'password', 'path', 'DDoS_type', 'DDoS_duration')
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True


class MaliciousClientForm(forms.ModelForm):
    class Meta:
        model = MaliciousClient
        fields = ('UE_id', 'hostname', 'ip', 'username', 'password', 'path')
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True


class AttackScenarioForm(forms.ModelForm):
    class Meta:
        model = AttackScenario
        fields = '__all__'
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True
        self.fields['mirai'].initial = True
        self.fields['ransomware'].initial = True
        self.fields['resource_hijacking'].initial = True
        self.fields['disk_wipe'].initial = True


class MachineLearningModelForm(forms.ModelForm):
    class Meta:
        model = MachineLearningModel
        fields = ("UE_id", "decision_tree", "naive_bayes", "extra_tree", "knn", "random_forest", "XGBoost")
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True


class SkipStageForm(forms.ModelForm):
    class Meta:
        model = SkipStage
        fields = {"UE_id", "skip_configuration", "skip_reproduction", "skip_data_processing", "skip_ML_training",
                  "skip_evaluation"}
        widgets = {'UE_id': forms.HiddenInput}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['UE_id'].disabled = True
