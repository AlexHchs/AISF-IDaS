# Future增加表單驗證失敗的欄位值保留
from django.shortcuts import render
from .models import Index, Testbed, Controller, DataLoggerServer, TargetServer, BenignServer, VulnerableClient, \
    NonVulnerableClient, AttackerServer, MaliciousClient, AttackScenario, ProgressData, MachineLearningModel, SkipStage
from .forms import IndexForm, TestbedForm, ControllerForm, DataLoggerServerForm, TargetServerForm, BenignServerForm, \
    VulnerableClientForm, NonVulnerableClientForm, AttackerServerForm, MaliciousClientForm, AttackScenarioForm, \
    MachineLearningModelForm, SkipStageForm
from django.shortcuts import redirect

from .serializers import ProgressDataSerializer
from rest_framework import viewsets

from .tasks import execute_toolchain
from django.contrib import messages
from django.forms import formset_factory
from celery import group
import os
import socket
import json
import pandas as pd

# Create your views here.

INDEX = 'index'
DASHBOARD = 'dashboard'
NEW_TESTBED = 'new_testbed'
NEW_TESTBED_INFORMATION = 'new_testbed_information'


# ---------- API ----------
class ProgressDataViewSet(viewsets.ModelViewSet):
    queryset = ProgressData.objects.all()
    serializer_class = ProgressDataSerializer
    # permission_classes = [permissions.IsAuthenticated]

def is_running_testbed():
    # check whether the testbed is running or not?
    testbeds = Testbed.objects.all()
    if testbeds:
        first_testbed = testbeds.first()
        if first_testbed.status == 2:
            return True

def not_exist_index():
    indexes = Index.objects.all()
    if not indexes:
        return True

def not_exist_testbed():
    testbeds = Testbed.objects.all()
    if not testbeds:
        return True

def create_progress_data_if_not_exist():
    progress_datas = ProgressData.objects.all()
    if not progress_datas:
        ProgressData.objects.create()

def validate_ips(hostname_ip_map):
    errors = []
    all_valid = True
    for hostname, ip in hostname_ip_map.items():
        if ' ' in ip:
            all_valid = False
            errors.append("({0}) {1} is not a valid IP address".format(hostname, ip))
        else:
            HOST_UP = False
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.settimeout(1)
                result = sock.connect_ex((ip, 22))
                if result == 0:
                    HOST_UP = True
                else:
                    HOST_UP = False
            except socket.error as exc:
                print("Caught exception socket.error : {0}".format(exc))
            finally:
                sock.close()

            if not HOST_UP:
                all_valid = False
                errors.append("Cannot connect with IP address {0} ({1})".format(ip, hostname))
    return all_valid, errors

def index(request):
    if is_running_testbed():
        return redirect(DASHBOARD)

    if request.method == "POST":
        form_index = IndexForm(request.POST)
        if form_index.is_valid():
            index = form_index.save(commit=False)
            index.number_of_UE = max(1, index.number_of_UE)
            index.save()
        return redirect(NEW_TESTBED)

    else:
        dict_forms = dict()
        dict_forms['Number of UEs:'] = IndexForm()
    return render(request, 'testbed/index.html', {'dict_forms': dict_forms})

def new_testbed(request):
    if not_exist_index():
        return redirect(INDEX)
    if is_running_testbed():
        return redirect(DASHBOARD)

    index = Index.objects.all().first()
    num_of_UE = index.number_of_UE

    if request.method == "POST":
        scenario_valid = []

        for i in range(num_of_UE):
            scenario_valid.append(False)

            form_testbed = TestbedForm(request.POST, prefix = 'tb_ue{0}'.format(i+1))
            if form_testbed.is_valid():
                form_testbed = form_testbed.save(commit=False)
                form_testbed.UE_id = i+1
                form_testbed.save()

            form_attack_scenario = AttackScenarioForm(request.POST, prefix = 'as_ue{0}'.format(i+1))
            if form_attack_scenario.is_valid():
                for field_name in form_attack_scenario.fields:
                    value = form_attack_scenario.cleaned_data[field_name]
                    scenario_valid[i] = scenario_valid[i] or value
                form_attack_scenario = form_attack_scenario.save(commit=False)
                form_attack_scenario.UE_id = i+1
                form_attack_scenario.save()

            form_machine_learning_model = MachineLearningModelForm(request.POST, prefix = 'ml_ue{0}'.format(i+1))
            if form_machine_learning_model.is_valid():
                form_machine_learning_model = form_machine_learning_model.save(commit=False)
                form_machine_learning_model.UE_id = i+1
                form_machine_learning_model.save()

            form_skip_stage = SkipStageForm(request.POST, prefix = 'ss_ue{0}'.format(i+1))
            if form_skip_stage.is_valid():
                form_skip_stage = form_skip_stage.save(commit=False)
                form_skip_stage.UE_id = i+1
                form_skip_stage.save()

        for i in range(num_of_UE):
            if not scenario_valid[i]:  # at least one scenario must be selected
                error = "UE-{0} Must select at least one attack scenario".format(i+1)
                messages.error(request, error)
                dict_forms = dict()
                for j in range(num_of_UE):
                    dict_forms['----------UE{0}----------'.format(j+1)] = ''
                    dict_forms['Number of machines (UE-{0}): '.format(j+1)] = TestbedForm(request.POST, prefix = 'ts_ue{0}'.format(j+1))
                    dict_forms['Scenario (UE-{0}):'.format(j+1)] = AttackScenarioForm(request.POST, prefix = 'as_ue{0}'.format(j+1))
                    dict_forms['Machine Learning model (UE-{0}):'.format(j+1)] = MachineLearningModelForm(request.POST, prefix = 'ml_ue{0}'.format(j+1))
                    dict_forms['Skip Stage (UE-{0}):'.format(j+1)] = SkipStageForm(request.POST, prefix = 'ss_ue{0}'.format(j+1))
                return render(request, 'testbed/new_testbed.html', {'dict_forms': dict_forms})

        return redirect(NEW_TESTBED_INFORMATION)

    else:
        dict_forms = dict()
        for i in range(num_of_UE):
            dict_forms['----------UE{0}----------'.format(i+1)] = ''
            dict_forms['Number of machines (UE-{0}): '.format(i+1)] = TestbedForm(request.POST, prefix = 'ts_ue{0}'.format(i+1))
            dict_forms['Scenario (UE-{0}):'.format(i+1)] = AttackScenarioForm(request.POST, prefix = 'as_ue{0}'.format(i+1))
            dict_forms['Machine Learning model (UE-{0}):'.format(i+1)] = MachineLearningModelForm(request.POST, prefix = 'ml_ue{0}'.format(i+1))
            dict_forms['Skip Stage (UE-{0}):'.format(i+1)] = SkipStageForm(request.POST, prefix = 'ss_ue{0}'.format(i+1))
    return render(request, 'testbed/new_testbed.html', {'dict_forms': dict_forms})

def new_testbed_information(request):
    if not_exist_testbed():
        return redirect(NEW_TESTBED)
    if is_running_testbed():
        return redirect(DASHBOARD)
    '''if SkipStage.skip_configuration and SkipStage.skip_reproduction:
        create_progress_data_if_not_exist()
        execute_toolchain.delay()
        return redirect(DASHBOARD)'''
        
    index = Index.objects.all().first()
    num_of_UE = index.number_of_UE
    testbed = Testbed.objects.all().first()

    if request.method == "POST":
        # clear all existing objects
        Controller.objects.all().delete()
        DataLoggerServer.objects.all().delete()
        TargetServer.objects.all().delete()
        BenignServer.objects.all().delete()
        VulnerableClient.objects.all().delete()
        NonVulnerableClient.objects.all().delete()
        AttackerServer.objects.all().delete()
        MaliciousClient.objects.all().delete()

        dict_hostname_ip = {}
        for i in range(num_of_UE):
            form_c = ControllerForm(request.POST, prefix='c_ue{0}'.format(i+1))
            if form_c.is_valid():
                form_c = form_c.save(commit=False)
                dict_hostname_ip[form_c.hostname] = form_c.ip
                form_c.UE_id = i+1
                form_c.save()

            form_dls = DataLoggerServerForm(request.POST, prefix='dls_ue{0}'.format(i+1))
            if form_dls.is_valid():
                form_dls = form_dls.save(commit=False)
                dict_hostname_ip[form_dls.hostname] = form_dls.ip
                form_dls.UE_id = i+1
                form_dls.save()

            form_ts = TargetServerForm(request.POST, prefix='ts_ue{0}'.format(i+1))
            if form_ts.is_valid():
                form_ts = form_ts.save(commit=False)
                dict_hostname_ip[form_ts.hostname] = form_ts.ip
                form_ts.UE_id = i+1
                form_ts.save()

            form_bs = BenignServerForm(request.POST, prefix='bs_ue{0}'.format(i+1))
            if form_bs.is_valid():
                form_bs = form_bs.save(commit=False)
                dict_hostname_ip[form_bs.hostname] = form_bs.ip
                form_bs.UE_id = i+1
                form_bs.save()

            for j in range(1):
                form_vc = VulnerableClientForm(request.POST, prefix='vc{0}_ue{1}'.format(j+1, i+1))
                if form_vc.is_valid():
                    form_vc = form_vc.save(commit=False)
                    dict_hostname_ip[form_vc.hostname] = form_vc.ip
                    form_vc.UE_id = i+1
                    form_vc.save()

            for j in range(2):
                form_nvc = NonVulnerableClientForm(request.POST, prefix='nvc{0}_ue{1}'.format(j+1, i+1))
                if form_nvc.is_valid():
                    form_nvc = form_nvc.save(commit=False)
                    dict_hostname_ip[form_nvc.hostname] = form_nvc.ip
                    form_nvc.UE_id = i+1
                    form_nvc.save()

            form_as = AttackerServerForm(request.POST, prefix='as_ue{0}'.format(i+1))
            if form_as.is_valid():
                attacker_server = form_as.save(commit=False)
                attacker_server.number_of_new_bots = 2
                dict_hostname_ip[attacker_server.hostname] = attacker_server.ip
                attacker_server.UE_id = i+1
                attacker_server.save()

            form_mc = MaliciousClientForm(request.POST, prefix='mc_ue{0}'.format(i+1))
            if form_mc.is_valid():
                form_mc = form_mc.save(commit=False)
                dict_hostname_ip[form_mc.hostname] = form_mc.ip
                form_mc.UE_id = i+1
                form_mc.save()

        # validate ip addresses
        all_valid, errors = validate_ips(dict_hostname_ip)
        if not all_valid:
            for error in errors:
                messages.error(request, error)

            dict_machines = dict()
            for i in range(num_of_UE):
                dict_machines['----------UE{0}----------'.format(i+1)] = ''
                dict_machines['UE-{0} Controller'.format(i+1)] = ControllerForm(request.POST, prefix='c_ue{0}'.format(i+1))
                dict_machines['UE-{0} Data Logger Server'.format(i+1)] = DataLoggerServerForm(request.POST, prefix='dls_ue{0}'.format(i+1))
                dict_machines['UE-{0} Target Server'.format(i+1)] = TargetServerForm(request.POST, prefix='ts_ue{0}'.format(i+1))
                dict_machines['UE-{0} Benign Server'.format(i+1)] = BenignServerForm(request.POST, prefix='bs_ue{0}'.format(i+1))
                for j in range(1):
                    dict_machines['UE-{0} Vulnerable Client {1}'.format(i+1, j+1)] = VulnerableClientForm(request.POST,prefix='vc{0}_ue{1}'.format(j+1, i+1))
                for j in range(2):
                    dict_machines['UE-{0} Non-Vulnerable Client {1}'.format(i+1, j+1)] = NonVulnerableClientForm(request.POST, prefix='nvc{0}_ue{1}'.format(j+1, i+1))
                dict_machines['UE-{0} Attacker Server'.format(i+1)] = AttackerServerForm(request.POST, prefix='as_ue{0}'.format(i+1))
                dict_machines['UE-{0} Malicious Client'.format(i+1)] = MaliciousClientForm(request.POST, prefix='mc_ue{0}'.format(i+1))
            return render(request, 'testbed/new_testbed_information.html', {'dict_machines': dict_machines})

        create_progress_data_if_not_exist()
        # execute_toolchain.delay()
        task_UEs_group = group(execute_toolchain(i) for i in range(num_of_UE))
        task_UEs_group()
        return redirect(DASHBOARD)

    else:
        dict_machines = dict()
        for i in range(num_of_UE):
            dict_machines['----------UE{0}----------'.format(i+1)] = ''
            dict_machines['UE-{0} Controller'.format(i+1)] = ControllerForm(request.POST, prefix='c_ue{0}'.format(i+1))
            dict_machines['UE-{0} Data Logger Server'.format(i+1)] = DataLoggerServerForm(request.POST, prefix='dls_ue{0}'.format(i+1))
            dict_machines['UE-{0} Target Server'.format(i+1)] = TargetServerForm(request.POST, prefix='ts_ue{0}'.format(i+1))
            dict_machines['UE-{0} Benign Server'.format(i+1)] = BenignServerForm(request.POST, prefix='bs_ue{0}'.format(i+1))
            for j in range(1):
                dict_machines['UE-{0} Vulnerable Client {1}'.format(i+1, j+1)] = VulnerableClientForm(request.POST,prefix='vc{0}_ue{1}'.format(j+1, i+1))
            for j in range(2):
                dict_machines['UE-{0} Non-Vulnerable Client {1}'.format(i+1, j+1)] = NonVulnerableClientForm(request.POST, prefix='nvc{0}_ue{1}'.format(j+1, i+1))
            dict_machines['UE-{0} Attacker Server'.format(i+1)] = AttackerServerForm(request.POST, prefix='as_ue{0}'.format(i+1))
            dict_machines['UE-{0} Malicious Client'.format(i+1)] = MaliciousClientForm(request.POST, prefix='mc_ue{0}'.format(i+1))
        return render(request, 'testbed/new_testbed_information.html', {'dict_machines': dict_machines})


def dashboard(request):
    create_progress_data_if_not_exist()
    testbeds = Testbed.objects.all()
    out = [[]]
    headers = []
    context = {}
    if testbeds:
        first_testbed = testbeds.first()
        if first_testbed.status == 3:
            # Ref : https://www.geeksforgeeks.org/rendering-data-frame-to-html-template-in-table-view-using-django-framework/
            file_dir = os.path.dirname(__file__)
            accuracy_dir = os.path.join(file_dir, '../CREME_backend_execution/evaluation_results/accuracy')
            data_sources = ['accounting', 'syslog', 'traffic']

            for i, source in enumerate(data_sources):
                data = []
                csv_path = os.path.join(accuracy_dir, 'accuracy_for_{}.csv'.format(source))
                df = pd.read_csv(csv_path) 
                json_records = df.reset_index().to_json(orient='records')
                data.append([])
                data = json.loads(json_records)
                
                context["d_{}".format(source)] = data

    return render(request, 'testbed/dashboard.html', context)