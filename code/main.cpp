/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:智联杯示例代码
 * Create: 2024/05/10
 */
#include <bits/stdc++.h>

using namespace std;
struct MessageTask {
    int msgType{};
    int usrInst{};
    int exeTime{};
    int deadLine{};
};
constexpr int MAX_USER_ID = 10000 + 5;

int n, m, c;
vector<vector<MessageTask>> tasks; // 每个用户任务列表
vector<vector<MessageTask>> cores; // 输出时，每个核的任务列表
int max_task_type_num = 201;

vector<int> user_task_index; // 用户下一个待调度的任务索引
vector<int> cores_time; // 核运行完所有任务的时间
vector<int> cores_task_type; // 分配给核的最后一个任务的类型
vector<int> user_all_time;  // 用户的任务时间总和
vector<vector<int>> users_of_core; // 每个核所负责的用户
vector<int> dispersed_users; // 没有对应核的用户
int all_q_score = 0, all_c_score = 0;
int all_exe_time = 0; // 所有任务运行时间总和
vector<vector<int>> core_task_type_num; // 核所负责的各个任务类型的任务数量（只计算用户待调度的第一个任务）
vector<int> dispersed_task_type_num; // 没有对应核的各个任务类型的任务数量（只计算用户待调度的第一个任务）
vector<int> cores_users_num; // 每个核的用户数量
vector<int> task_type_num_of_core; // 统计每个核最后一个任务类型数量
int min_user_time = INT_MAX;
int max_user_time = INT_MIN;

void get_users_of_core(){
    for (int uid = 0; uid < MAX_USER_ID; ++uid) {
        if (tasks[uid].empty()) {
            continue;
        }
        auto min_iter = min_element(cores_time.begin(), cores_time.end());
        int min_index = distance(cores_time.begin(), min_iter);

        users_of_core[min_index].push_back(uid);
        cores_time[min_index] += user_all_time[uid];
    }
}

bool is_change(int select_user, int select_q_s, int select_c_s, int q_s, int c_s, MessageTask task, int core_id){
    if(select_user == -1) return true;
    MessageTask select_task = tasks[select_user][user_task_index[select_user]];
    if(select_q_s + select_c_s < q_s + c_s) return true;
    if(select_q_s == 0 && select_c_s == 1 && q_s == 1 && c_s == 0) return true;
    if(select_q_s == q_s && select_c_s == c_s){  // 根据任务类型和任务执行时间
        if(task.msgType != select_task.msgType){
            if((core_task_type_num[core_id][task.msgType] + dispersed_task_type_num[task.msgType]) / (task_type_num_of_core[task.msgType] + 1) > \
            core_task_type_num[core_id][select_task.msgType] + dispersed_task_type_num[select_task.msgType] / (task_type_num_of_core[select_task.msgType] + 1)) return true;
        }
        else{
            if(task.deadLine - task.exeTime < select_task.deadLine - select_task.exeTime) return true;
        }
    }
    return false;
}

void demo(){
    // 3.简单调度逻辑：每次选定一个用户，轮替选定一个核，将用户对应的所有任务放入该核中调度，尽可能用户均衡
    // 输出的调度需要满足两个约束：1.用户的任务需要按顺序依次执行；2.同一个用户的任务必须放到同一个核上
    int curChooseCore = 0;
    for (int uid = 0; uid < MAX_USER_ID; ++uid) {
        if (tasks[uid].empty()) {
            continue;
        }
        auto &core = cores[curChooseCore];
        for (auto &task : tasks[uid]) {
            core.push_back(task);
        }
        curChooseCore++;
        curChooseCore = curChooseCore % m;
    }
}

void demo_1(){
    get_users_of_core();
    for (int core_id = 0; core_id < m; ++core_id) {
        for (int i = 0; i < users_of_core[core_id].size(); i ++){
            for (auto &task : tasks[users_of_core[core_id][i]]) {
                cores[core_id].push_back(task);
            }
        }
    }
}

void demo_2(){
    get_users_of_core();
    for (int core_id = 0; core_id < m; ++core_id) {  
        while(true){
            int task_id = -1;
            int user_id;
            for (int i = 0; i < users_of_core[core_id].size(); i ++){
                int uid = users_of_core[core_id][i];
                if(user_task_index[uid] < tasks[uid].size()){
                    user_id = uid;
                    task_id = user_task_index[uid];
                    if(cores_task_type[core_id] == -1 || tasks[uid][user_task_index[uid]].msgType == cores_task_type[core_id]) break;
                }
            }
            if(task_id == -1){
                break;
            }
            else{
                cores[core_id].push_back(tasks[user_id][task_id]);
                cores_task_type[core_id] = tasks[user_id][task_id].msgType;
                user_task_index[user_id] ++;
            }
        }
    }
}

void demo_3(){
    int task_num = 0; // 已分配任务数量
    while(task_num < n){

        auto min_iter = min_element(cores_time.begin(), cores_time.end());
        int min_index = distance(cores_time.begin(), min_iter);  // 选择运行时间最少的核，为其分配任务 (users_of_core[min_index], dispersed_users)中选择用户的任务

        int select_q_s = 0, select_c_s = 0; // 选择对应任务的亲和分和任务完成分  
        int select_user = -1;
        for(int i = 0; i < users_of_core[min_index].size(); i ++){  // 从负责的用户集中选择
            int uid = users_of_core[min_index][i];
            if(user_task_index[uid] < tasks[uid].size()){
                int q_s = 0, c_s = 0;
                MessageTask task = tasks[uid][user_task_index[uid]];
                if(cores_task_type[min_index] == task.msgType) q_s = 1;
                if(task.exeTime + cores_time[min_index] <= task.deadLine) c_s = 1;
                if(select_user == -1 || (select_q_s + select_c_s < q_s + c_s) || (select_q_s == 0 && select_c_s == 1 && q_s == 1 && c_s == 0) ||\
                (select_q_s == q_s && select_c_s == c_s && task.exeTime < tasks[select_user][user_task_index[select_user]].exeTime)){
                    select_user = uid;
                    select_q_s = q_s;
                    select_c_s = c_s;
                }
            }
        }
        for(int i = 0; i < dispersed_users.size(); i ++){  // 从未分配的用户集中选
            int uid = dispersed_users[i];
            if(user_task_index[uid] < tasks[uid].size()){
                int q_s = 0, c_s = 0;
                MessageTask task = tasks[uid][user_task_index[uid]];
                if(cores_task_type[min_index] == task.msgType) q_s = 1;
                if(task.exeTime + cores_time[min_index] <= task.deadLine) c_s = 1;
                if(select_user == -1 || (select_q_s + select_c_s < q_s + c_s) || (select_q_s == 0 && select_c_s == 1 && q_s == 1 && c_s == 0) ||\
                (select_q_s == q_s && select_c_s == c_s && task.exeTime < tasks[select_user][user_task_index[select_user]].exeTime)){
                    select_user = uid;
                    select_q_s = q_s;
                    select_c_s = c_s;
                }
            }
        }
        if(select_user != -1){  // 选择成功
            cores[min_index].push_back(tasks[select_user][user_task_index[select_user]]);

            cores_task_type[min_index] = tasks[select_user][user_task_index[select_user]].msgType;
            cores_time[min_index] += tasks[select_user][user_task_index[select_user]].exeTime;

            task_num ++;
            user_task_index[select_user] ++;

            all_q_score += select_q_s;
            all_c_score += select_c_s;
        }  // 没有可选任务
        else cores_time[min_index] = INT_MAX;

        auto position = find(dispersed_users.begin(), dispersed_users.end(), select_user);  // 核所负责用户集和未分配用户集 两个集合的变动
        if(position != dispersed_users.end()){
            dispersed_users.erase(position);
            users_of_core[min_index].push_back(select_user);
        }
        if(select_user != -1 && user_task_index[select_user] == tasks[select_user].size()) users_of_core[min_index].erase(find(users_of_core[min_index].begin(), users_of_core[min_index].end(), select_user));
    }
}

void demo_4(){
    int task_num = 0; // 已分配任务数量
    core_task_type_num = vector<vector<int>>(m, vector<int>(max_task_type_num, 0));
    vector<int> core_all_time = vector<int>(m, 0);
    for(int i = 0; i < MAX_USER_ID; i ++){
        if(user_all_time[i]) min_user_time = min(min_user_time, user_all_time[i]);
        if(user_all_time[i]) max_user_time = max(max_user_time, user_all_time[i]);
    }
    while(task_num < n){

        auto min_iter = min_element(cores_time.begin(), cores_time.end());
        int min_index = distance(cores_time.begin(), min_iter);  // 选择运行时间最少的核，为其分配任务 (users_of_core[min_index], dispersed_users)中选择用户的任务

        int select_q_s = 0, select_c_s = 0; // 选择对应任务的亲和分和任务完成分  
        int select_user = -1;
        for(int i = 0; i < users_of_core[min_index].size(); i ++){  // 从负责的用户集中选择
            int uid = users_of_core[min_index][i];
            if(user_task_index[uid] < tasks[uid].size()){
                int q_s = 0, c_s = 0;
                MessageTask task = tasks[uid][user_task_index[uid]];
                if(cores_task_type[min_index] == task.msgType) q_s = 1;
                if(task.exeTime + cores_time[min_index] <= task.deadLine) c_s = 1;
                if(is_change(select_user, select_q_s, select_c_s, q_s, c_s, task, min_index)){
                    select_user = uid;
                    select_q_s = q_s;
                    select_c_s = c_s;
                }
            }
        }
        for(int i = 0; i < dispersed_users.size(); i ++){  // 从未分配的用户集中选
            int uid = dispersed_users[i];
            if(user_task_index[uid] < tasks[uid].size()){
                int q_s = 0, c_s = 0;
                MessageTask task = tasks[uid][user_task_index[uid]];
                if(cores_task_type[min_index] == task.msgType) q_s = 1;
                if(task.exeTime + cores_time[min_index] <= task.deadLine) c_s = 1;
                if(is_change(select_user, select_q_s, select_c_s, q_s, c_s, task, min_index)){
                    select_user = uid;
                    select_q_s = q_s;
                    select_c_s = c_s;
                }
            }
        }
        if(select_user != -1){  // 选择成功
            cores[min_index].push_back(tasks[select_user][user_task_index[select_user]]);
            cores_users_num[min_index] ++;
            task_type_num_of_core[cores_task_type[min_index]] --;
            task_type_num_of_core[tasks[select_user][user_task_index[select_user]].msgType] ++;

            cores_task_type[min_index] = tasks[select_user][user_task_index[select_user]].msgType;
            cores_time[min_index] += tasks[select_user][user_task_index[select_user]].exeTime;

            task_num ++;
            user_task_index[select_user] ++;

            all_q_score += select_q_s;
            all_c_score += select_c_s;
        }  // 没有可选任务
        else cores_time[min_index] = INT_MAX;

        auto position = find(dispersed_users.begin(), dispersed_users.end(), select_user);  // 核所负责用户集和未分配用户集 两个集合的变动
        if(position != dispersed_users.end()){
            dispersed_users.erase(position);
            users_of_core[min_index].push_back(select_user);
            core_all_time[min_index] += user_all_time[select_user];

            dispersed_task_type_num[tasks[select_user][user_task_index[select_user] - 1].msgType] --;
        }
        else if (select_user != -1) {
            core_task_type_num[min_index][tasks[select_user][user_task_index[select_user] - 1].msgType] --;
        }
        if(select_user != -1 && user_task_index[select_user] == tasks[select_user].size()) {
            users_of_core[min_index].erase(find(users_of_core[min_index].begin(), users_of_core[min_index].end(), select_user));
        }
        else if (select_user != -1){
            core_task_type_num[min_index][tasks[select_user][user_task_index[select_user]].msgType] ++;
        }
    }
}

int main() 
{
    // 1.读取任务数、核数、系统最大执行时间
    scanf("%d %d %d", &n, &m, &c);

    cores = vector<vector<MessageTask>>(m);

    user_task_index = vector<int>(MAX_USER_ID, 0);
    cores_time = vector<int>(m, 0);
    cores_task_type = vector<int>(m, -1);
    user_all_time = vector<int>(MAX_USER_ID, 0);
    users_of_core = vector<vector<int>>(m);
    dispersed_task_type_num = vector<int>(max_task_type_num, 0);
    cores_users_num = vector<int> (m, 0);
    task_type_num_of_core = vector<int>(max_task_type_num, 0);

    // 2.读取每个任务的信息
    tasks.resize(MAX_USER_ID);
    for (int i = 1; i <= n; i++) {
        MessageTask task;
        scanf("%d %d %d %d", &task.msgType, &task.usrInst, &task.exeTime, &task.deadLine);
        task.deadLine = std::min(task.deadLine, c);
        tasks[task.usrInst].push_back(task);
        user_all_time[task.usrInst] += task.exeTime;
        all_exe_time += task.exeTime;

        if(tasks[task.usrInst].size() == 1){
            dispersed_users.push_back(task.usrInst);  
            dispersed_task_type_num[task.msgType] ++;
        }
    }
    
    // 3.逻辑调度
    demo_4();

    // 4.输出结果，使用字符串存储，一次IO输出
    int q_score = 0;
    int c_score = 0;
    cores_time = vector<int>(m, 0);
    stringstream out;
    for (int coreId = 0; coreId < m; ++coreId) {
        out << cores[coreId].size();
        int task_type = -1;
        for (auto &task : cores[coreId]) {
            out << " " << task.msgType << " " << task.usrInst;
            if(task_type == task.msgType) q_score ++;
            task_type = task.msgType;
            if(task.deadLine >= task.exeTime + cores_time[coreId]) c_score ++;
            cores_time[coreId] += task.exeTime;
        }
        out << endl;
    }
    printf("%s", out.str().c_str());
    for (int coreId = 0; coreId < m; ++coreId) cout << cores_time[coreId] << " ";
    cout << endl;
    for (int coreId = 0; coreId < m; ++coreId) cout << cores_users_num[coreId] << " ";
    cout << endl;
    for (int coreId = 0; coreId < m; ++coreId) cout << cores_time[coreId] / cores_users_num[coreId] << " ";
    cout << endl;
    cout << min_user_time << " " << max_user_time << endl;
    cout << "q_score:" << q_score << " c_score:" << c_score << " q_score + c_score:" << q_score + c_score << endl;
    cout << "all_q_score:" << all_q_score << " all_c_score:" << all_c_score << " all_q_score + all_c_score:" << all_q_score + all_c_score;
    return 0;
}