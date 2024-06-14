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
vector<vector<int>> user_task_type_num; // 每个用户的每个任务类型的任务数量
vector<int> user_max_task_type; // 每个用户最多的任务类型
vector<int> user_max_task_type_num; // 每个用户最多的任务类型的任务数量
vector<vector<int>> task_type_user_list; // 每个任务类型任务数量最多的用户列表

vector<int> init_dispersed_users; // 没有对应核的用户
vector<int> init_dispersed_task_type_num; // 没有对应核的各个任务类型的任务数量（只计算用户待调度的第一个任务）
int result_score;
vector<vector<MessageTask>> result_cores; // 输出时，每个核的任务列表


void init(){
    dispersed_users = init_dispersed_users;  
    dispersed_task_type_num = init_dispersed_task_type_num;

    cores = vector<vector<MessageTask>>(m);
    user_task_index = vector<int>(MAX_USER_ID, 0);
    cores_time = vector<int>(m, 0);
    cores_task_type = vector<int>(m, -1);
    users_of_core = vector<vector<int>>(m);
    cores_users_num = vector<int> (m, 0);
    task_type_num_of_core = vector<int>(max_task_type_num, 0);

    all_c_score = 0;
    all_q_score = 0;
}

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
    dispersed_users = vector<int>();  
    dispersed_task_type_num = vector<int>(max_task_type_num, 0);
}

bool is_change(int select_user, int select_q_s, int select_c_s, int select_next_q, int q_s, int c_s, int next_q, MessageTask task, int core_id){
    if(select_user == -1) return true;
    MessageTask select_task = tasks[select_user][user_task_index[select_user]];
    if(select_q_s + select_c_s < q_s + c_s) return true;
    if(select_q_s == 0 && select_c_s == 1 && q_s == 1 && c_s == 0) return true;
    if(select_q_s == q_s && select_c_s == c_s){  // 根据任务类型和任务执行时间
        if(task.msgType != select_task.msgType){
            if(core_task_type_num[core_id][task.msgType] + dispersed_task_type_num[task.msgType] / (task_type_num_of_core[task.msgType] + 1) > \
            core_task_type_num[core_id][select_task.msgType] + dispersed_task_type_num[select_task.msgType] / (task_type_num_of_core[select_task.msgType] + 1)) return true;
        }
        else{
            if(task.deadLine - task.exeTime < select_task.deadLine - select_task.exeTime) return true;
        }
        // if(next_q > select_next_q) return true;
    }
    return false;
}

bool is_change_demo5(int select_user, int select_q_s, int select_next_q, int select_c_s, int q_s, int next_q, int c_s, int u_id, int core_id){
    if(select_user == -1) return true;
    MessageTask task = tasks[u_id][user_task_index[u_id]];
    MessageTask select_task = tasks[select_user][user_task_index[select_user]];
    // if(select_q_s < q_s) return true;
    if(select_q_s + select_c_s < q_s + c_s) return true;
    if(select_q_s == 0 && select_c_s == 1 && q_s == 1 && c_s == 0) return true;
    if(select_q_s == q_s && select_c_s == c_s){  // 根据任务类型和任务执行时间
        if(next_q > select_next_q) return true;
        if(task.msgType != select_task.msgType){
            if(core_task_type_num[core_id][task.msgType] + dispersed_task_type_num[task.msgType] / (task_type_num_of_core[task.msgType] + 1) > \
            core_task_type_num[core_id][select_task.msgType] + dispersed_task_type_num[select_task.msgType] / (task_type_num_of_core[select_task.msgType] + 1)) return true;
        }
        else{
            if(task.deadLine - task.exeTime < select_task.deadLine - select_task.exeTime) return true;
        }
    }
    return false;
}

bool is_change_demo6(int select_user, int select_q_s, int select_next_q, int select_c_s, int q_s, int next_q, int c_s, int u_id, int core_id, bool flag){
    if(select_user == -1) return true;
    int disperset_num = 0;
    MessageTask task = tasks[u_id][user_task_index[u_id]];
    MessageTask select_task = tasks[select_user][user_task_index[select_user]];
    if(select_q_s + select_c_s < q_s + c_s) return true;
    if(select_q_s == 0 && select_c_s == 1 && q_s == 1 && c_s == 0) return true;
    if(select_q_s == q_s && select_c_s == c_s){  // 根据任务类型和任务执行时间
        if(next_q > select_next_q) return true;
        if(task.msgType != select_task.msgType){
            if(flag) disperset_num = dispersed_task_type_num[task.msgType];
            if(core_task_type_num[core_id][task.msgType] + disperset_num / (task_type_num_of_core[task.msgType] + 1) > \
            core_task_type_num[core_id][select_task.msgType] + disperset_num / (task_type_num_of_core[select_task.msgType] + 1)) return true;
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
                // if(select_user == -1 || (select_q_s + select_c_s < q_s + c_s) || (select_q_s == 0 && select_c_s == 1 && q_s == 1 && c_s == 0) ||\
                // (select_q_s == q_s && select_c_s == c_s && task.exeTime < tasks[select_user][user_task_index[select_user]].exeTime)){
                //     select_user = uid;
                //     select_q_s = q_s;
                //     select_c_s = c_s;
                // }
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
    init();
    int task_num = 0; // 已分配任务数量
    core_task_type_num = vector<vector<int>>(m, vector<int>(max_task_type_num, 0));
    vector<int> core_all_time = vector<int>(m, 0);
    for(int i = 0; i < MAX_USER_ID; i ++){
        if(user_all_time[i] != 0) min_user_time = min(min_user_time, user_all_time[i]);
        if(user_all_time[i] != 0) max_user_time = max(max_user_time, user_all_time[i]);
    }
    while(task_num < n){

        auto min_iter = min_element(cores_time.begin(), cores_time.end());
        int min_index = distance(cores_time.begin(), min_iter);  // 选择运行时间最少的核，为其分配任务 (users_of_core[min_index], dispersed_users)中选择用户的任务

        int select_q_s = 0, select_c_s = 0; // 选择对应任务的亲和分和任务完成分  
        int select_next_q = 0;
        int select_user = -1;
        for(int i = 0; i < users_of_core[min_index].size(); i ++){  // 从负责的用户集中选择
            int uid = users_of_core[min_index][i];
            if(user_task_index[uid] < tasks[uid].size()){
                int q_s = 0, c_s = 0;
                int next_q;
                if(user_task_index[uid] + 1 == tasks[uid].size()) next_q = 0;
                else next_q = core_task_type_num[min_index][tasks[uid][user_task_index[uid] + 1].msgType] + dispersed_task_type_num[tasks[uid][user_task_index[uid] + 1].msgType];
                MessageTask task = tasks[uid][user_task_index[uid]];
                if(cores_task_type[min_index] == task.msgType) q_s = 1;
                if(task.exeTime + cores_time[min_index] <= task.deadLine) c_s = 1;
                if(is_change(select_user, select_q_s, select_c_s, select_next_q, q_s, c_s, next_q, task, min_index)){
                    select_user = uid;
                    select_q_s = q_s;
                    select_c_s = c_s;
                    select_next_q = next_q;
                }
            }
        }
        for(int i = 0; i < dispersed_users.size(); i ++){  // 从未分配的用户集中选
            int uid = dispersed_users[i];
            if(user_task_index[uid] < tasks[uid].size()){
                int q_s = 0, c_s = 0;
                int next_q;
                if(user_task_index[uid] + 1 == tasks[uid].size()) next_q = 0;
                else next_q = core_task_type_num[min_index][tasks[uid][user_task_index[uid] + 1].msgType] + dispersed_task_type_num[tasks[uid][user_task_index[uid] + 1].msgType];
                MessageTask task = tasks[uid][user_task_index[uid]];
                if(cores_task_type[min_index] == task.msgType) q_s = 1;
                if(task.exeTime + cores_time[min_index] <= task.deadLine) c_s = 1;
                if(is_change(select_user, select_q_s, select_c_s, select_next_q, q_s, c_s, next_q, task, min_index)){
                    select_user = uid;
                    select_q_s = q_s;
                    select_c_s = c_s;
                    select_next_q = next_q;
                }
            }
        }
        if(select_user != -1){  // 选择成功
            cores[min_index].push_back(tasks[select_user][user_task_index[select_user]]);
            if(cores_task_type[min_index] != -1) task_type_num_of_core[cores_task_type[min_index]] --;
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
            cores_users_num[min_index] ++;

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
    
    if(all_c_score + all_q_score > result_score){
        result_score = all_c_score + all_q_score;
        result_cores = cores;
    }
}

void demo_5(){
    int task_num = 0; // 已分配任务数量
    core_task_type_num = vector<vector<int>>(m, vector<int>(max_task_type_num, 0));
    vector<int> core_all_time = vector<int>(m, 0);
    for(int i = 0; i < MAX_USER_ID; i ++){
        if(user_all_time[i] != 0) min_user_time = min(min_user_time, user_all_time[i]);
        if(user_all_time[i] != 0) max_user_time = max(max_user_time, user_all_time[i]);
    }
    
    int min_index = 0;
    while(task_num < n){

        // auto min_iter = min_element(cores_time.begin(), cores_time.end());
        // int min_index = distance(cores_time.begin(), min_iter);  // 选择运行时间最少的核，为其分配任务 (users_of_core[min_index], dispersed_users)中选择用户的任务

        int select_q_s = 0, select_c_s = 0; // 选择对应任务的亲和分和任务完成分  
        int select_next_q = 0; // 选择对应任务下一个任务能否获得亲和分
        int select_user = -1;
        for(int i = 0; i < users_of_core[min_index].size(); i ++){  // 从负责的用户集中选择
            int uid = users_of_core[min_index][i];
            if(user_task_index[uid] < tasks[uid].size()){
                int q_s = 0, c_s = 0;
                int next_q;
                if(user_task_index[uid] + 1 == tasks[uid].size()) next_q = 0;
                else next_q = core_task_type_num[min_index][tasks[uid][user_task_index[uid] + 1].msgType] + dispersed_task_type_num[tasks[uid][user_task_index[uid] + 1].msgType];
                MessageTask task = tasks[uid][user_task_index[uid]];
                if(cores_task_type[min_index] == task.msgType) q_s = 1;
                if(task.exeTime + cores_time[min_index] <= task.deadLine) c_s = 1;
                // cout << min_index << " " << uid << " " << q_s << " " << c_s << " " << next_q << endl;
                if(is_change_demo5(select_user, select_q_s, select_next_q, select_c_s, q_s, next_q, c_s, uid, min_index)){
                    select_user = uid;
                    select_q_s = q_s;
                    select_c_s = c_s;
                    select_next_q = next_q;
                }
            }
        }
        if(core_all_time[min_index] < all_exe_time / m){
            for(int i = 0; i < dispersed_users.size(); i ++){  // 从未分配的用户集中选
                int uid = dispersed_users[i];
                if(user_task_index[uid] < tasks[uid].size()){
                    int q_s = 0, c_s = 0;
                    int next_q;
                    if(user_task_index[uid] + 1 == tasks[uid].size()) next_q = 0;
                    else next_q = core_task_type_num[min_index][tasks[uid][user_task_index[uid] + 1].msgType] + dispersed_task_type_num[tasks[uid][user_task_index[uid] + 1].msgType];
                    MessageTask task = tasks[uid][user_task_index[uid]];
                    if(cores_task_type[min_index] == task.msgType) q_s = 1;
                    if(task.exeTime + cores_time[min_index] <= task.deadLine) c_s = 1;
                    // cout << min_index << " " << uid << " " << q_s << " " << c_s << " " << next_q << endl;
                    if(is_change_demo5(select_user, select_q_s, select_next_q, select_c_s, q_s, next_q, c_s, uid, min_index)){
                        select_user = uid;
                        select_q_s = q_s;
                        select_c_s = c_s;
                        select_next_q = next_q;
                    }
                }
            }
        }
        if(select_user != -1){  // 选择成功
            cores[min_index].push_back(tasks[select_user][user_task_index[select_user]]);
            if(cores_task_type[min_index] != -1) task_type_num_of_core[cores_task_type[min_index]] --;
            task_type_num_of_core[tasks[select_user][user_task_index[select_user]].msgType] ++;

            cores_task_type[min_index] = tasks[select_user][user_task_index[select_user]].msgType;
            cores_time[min_index] += tasks[select_user][user_task_index[select_user]].exeTime;

            task_num ++;
            user_task_index[select_user] ++;

            all_q_score += select_q_s;
            all_c_score += select_c_s;
        }  // 没有可选任务
        else min_index += 1;

        auto position = find(dispersed_users.begin(), dispersed_users.end(), select_user);  // 核所负责用户集和未分配用户集 两个集合的变动
        if(position != dispersed_users.end()){
            dispersed_users.erase(position);
            users_of_core[min_index].push_back(select_user);
            core_all_time[min_index] += user_all_time[select_user];
            cores_users_num[min_index] ++;

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

void demo_6(){
    init();

    int task_num = 0; // 已分配任务数量
    core_task_type_num = vector<vector<int>>(m, vector<int>(max_task_type_num, 0));
    vector<int> core_all_time = vector<int>(m, 0);
    for(int i = 0; i < MAX_USER_ID; i ++){
        if(user_all_time[i] != 0) min_user_time = min(min_user_time, user_all_time[i]);
        if(user_all_time[i] != 0) max_user_time = max(max_user_time, user_all_time[i]);
    }
    
    int min_index = 0;
    while(task_num < n){

        // auto min_iter = min_element(cores_time.begin(), cores_time.end());
        // int min_index = distance(cores_time.begin(), min_iter);  // 选择运行时间最少的核，为其分配任务 (users_of_core[min_index], dispersed_users)中选择用户的任务

        int select_q_s = 0, select_c_s = 0; // 选择对应任务的亲和分和任务完成分  
        int select_next_q = 0; // 选择对应任务下一个任务能否获得亲和分
        int select_user = -1;
        for(int i = 0; i < users_of_core[min_index].size(); i ++){  // 从负责的用户集中选择
            int uid = users_of_core[min_index][i];
            if(user_task_index[uid] < tasks[uid].size()){
                int q_s = 0, c_s = 0;
                int next_q;
                if(user_task_index[uid] + 1 == tasks[uid].size()) next_q = 0;
                else if(core_all_time[min_index] < all_exe_time / m) next_q = core_task_type_num[min_index][tasks[uid][user_task_index[uid] + 1].msgType] + dispersed_task_type_num[tasks[uid][user_task_index[uid] + 1].msgType];
                else next_q = core_task_type_num[min_index][tasks[uid][user_task_index[uid] + 1].msgType];
                MessageTask task = tasks[uid][user_task_index[uid]];
                if(cores_task_type[min_index] == task.msgType) q_s = 1;
                if(task.exeTime + cores_time[min_index] <= task.deadLine) c_s = 1;
                // cout << min_index << " " << uid << " " << q_s << " " << c_s << " " << next_q << endl;
                if(is_change_demo6(select_user, select_q_s, select_next_q, select_c_s, q_s, next_q, c_s, uid, min_index, core_all_time[min_index] >= all_exe_time / m)){
                    select_user = uid;
                    select_q_s = q_s;
                    select_c_s = c_s;
                    select_next_q = next_q;
                }
            }
        }
        if(core_all_time[min_index] < all_exe_time / m){
            for(int i = 0; i < dispersed_users.size(); i ++){  // 从未分配的用户集中选
                int uid = dispersed_users[i];
                if(user_task_index[uid] < tasks[uid].size()){
                    int q_s = 0, c_s = 0;
                    int next_q;
                    if(user_task_index[uid] + 1 == tasks[uid].size()) next_q = 0;
                    else next_q = core_task_type_num[min_index][tasks[uid][user_task_index[uid] + 1].msgType] + dispersed_task_type_num[tasks[uid][user_task_index[uid] + 1].msgType];
                    MessageTask task = tasks[uid][user_task_index[uid]];
                    if(cores_task_type[min_index] == task.msgType) q_s = 1;
                    if(task.exeTime + cores_time[min_index] <= task.deadLine) c_s = 1;
                    // cout << min_index << " " << uid << " " << q_s << " " << c_s << " " << next_q << endl;
                    if(is_change_demo6(select_user, select_q_s, select_next_q, select_c_s, q_s, next_q, c_s, uid, min_index, true)){
                        select_user = uid;
                        select_q_s = q_s;
                        select_c_s = c_s;
                        select_next_q = next_q;
                    }
                }
            }
        }
        if(select_user != -1){  // 选择成功
            cores[min_index].push_back(tasks[select_user][user_task_index[select_user]]);
            if(cores_task_type[min_index] != -1) task_type_num_of_core[cores_task_type[min_index]] --;
            task_type_num_of_core[tasks[select_user][user_task_index[select_user]].msgType] ++;

            cores_task_type[min_index] = tasks[select_user][user_task_index[select_user]].msgType;
            cores_time[min_index] += tasks[select_user][user_task_index[select_user]].exeTime;

            task_num ++;
            user_task_index[select_user] ++;

            all_q_score += select_q_s;
            all_c_score += select_c_s;
        }  // 没有可选任务
        else min_index += 1;

        auto position = find(dispersed_users.begin(), dispersed_users.end(), select_user);  // 核所负责用户集和未分配用户集 两个集合的变动
        if(position != dispersed_users.end()){
            dispersed_users.erase(position);
            users_of_core[min_index].push_back(select_user);
            core_all_time[min_index] += user_all_time[select_user];
            cores_users_num[min_index] ++;

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

    if(all_c_score + all_q_score > result_score){
        result_score = all_c_score + all_q_score;
        result_cores = cores;
    }

    
}


int main() 
{
    // 1.读取任务数、核数、系统最大执行时间
    scanf("%d %d %d", &n, &m, &c);  

    user_all_time = vector<int>(MAX_USER_ID, 0);
    result_score = 0;

    cores = vector<vector<MessageTask>>(m);
    user_task_index = vector<int>(MAX_USER_ID, 0);
    cores_time = vector<int>(m, 0);
    cores_task_type = vector<int>(m, -1);
    users_of_core = vector<vector<int>>(m);
    dispersed_task_type_num = vector<int>(max_task_type_num, 0);
    cores_users_num = vector<int> (m, 0);
    task_type_num_of_core = vector<int>(max_task_type_num, 0);

    user_task_type_num = vector<vector<int>>(10001, vector<int>(201, 0));
    user_max_task_type = vector<int>(10001, -1);
    user_max_task_type_num = vector<int>(10001, 0);
    task_type_user_list = vector<vector<int>>(201, vector<int>());

    init_dispersed_task_type_num = vector<int>(max_task_type_num, 0);

    // 2.读取每个任务的信息
    tasks.resize(MAX_USER_ID);
    for (int i = 1; i <= n; i++) {
        MessageTask task;
        scanf("%d %d %d %d", &task.msgType, &task.usrInst, &task.exeTime, &task.deadLine);
        task.deadLine = std::min(task.deadLine, c);
        // task.exeTime = task.exeTime * 6 / 5;
        tasks[task.usrInst].push_back(task);
        user_all_time[task.usrInst] += task.exeTime;
        all_exe_time += task.exeTime;

        user_task_type_num[task.usrInst][task.msgType] ++;
        if (user_task_type_num[task.usrInst][task.msgType] > user_max_task_type_num[task.usrInst]){
            user_max_task_type[task.usrInst] = task.msgType;
            user_max_task_type_num[task.usrInst] = user_task_type_num[task.usrInst][task.msgType];
        }

        if(tasks[task.usrInst].size() == 1){
            init_dispersed_users.push_back(task.usrInst);  
            init_dispersed_task_type_num[task.msgType] ++;
        }
    }


    
    // 3.逻辑调度
    demo_4();
    demo_6();

    // 4.输出结果，使用字符串存储，一次IO输出
    int q_score = 0;
    int c_score = 0;
    cores_time = vector<int>(m, 0);
    stringstream out;
    for (int coreId = 0; coreId < m; ++coreId) {
        out << result_cores[coreId].size();
        int task_type = -1;
        for (auto &task : result_cores[coreId]) {
            out << " " << task.msgType << " " << task.usrInst;
            if(task_type == task.msgType) q_score ++;
            task_type = task.msgType;
            // task.exeTime = task.exeTime * 6 / 5;
            if(task.deadLine >= task.exeTime + cores_time[coreId]) c_score ++;
            cores_time[coreId] += task.exeTime;
        }
        out << endl;
    }
    printf("%s", out.str().c_str());
    // for (int coreId = 0; coreId < m; ++coreId) cout << cores_time[coreId] << " ";
    // cout << endl;
    // for (int coreId = 0; coreId < m; ++coreId) cout << cores_users_num[coreId] << " ";
    // cout << endl;
    // for (int coreId = 0; coreId < m; ++coreId)
    //     if(cores_users_num[coreId] != 0) cout << cores_time[coreId] / cores_users_num[coreId] << " ";
    //     else cout << 0 << " ";
    // cout << endl;
    // cout << min_user_time << " " << max_user_time << endl;
    // cout << "q_score:" << q_score << " c_score:" << c_score << " q_score + c_score:" << q_score + c_score << endl;
    // cout << "all_q_score:" << all_q_score << " all_c_score:" << all_c_score << " all_q_score + all_c_score:" << all_q_score + all_c_score;
    return 0;
}