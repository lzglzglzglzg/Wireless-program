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

vector<int> user_task_index; // 用户下一个待调度的任务索引
vector<int> cores_time; // 核运行完所有任务的时间
vector<int> cores_task_type; // 分配给核的最后一个任务的类型
vector<int> user_all_time;  // 用户的任务时间总和
vector<vector<int>> users_of_core; // 每个核所负责的用户

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
    for (int core_id = 0; core_id < m; ++core_id) {
        for (int i = 0; i < users_of_core[core_id].size(); i ++){
            for (auto &task : tasks[users_of_core[core_id][i]]) {
                cores[core_id].push_back(task);
            }
        }
    }
}

void demo_2(){
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

    // 2.读取每个任务的信息
    tasks.resize(MAX_USER_ID);
    for (int i = 1; i <= n; i++) {
        MessageTask task;
        scanf("%d %d %d %d", &task.msgType, &task.usrInst, &task.exeTime, &task.deadLine);
        task.deadLine = std::min(task.deadLine, c);
        tasks[task.usrInst].push_back(task);
        user_all_time[task.usrInst] += task.exeTime;
    }
    get_users_of_core();
    
    // 3.逻辑调度
    demo_1();

    // 4.输出结果，使用字符串存储，一次IO输出
    stringstream out;
    for (int coreId = 0; coreId < m; ++coreId) {
        out << cores[coreId].size();
        for (auto &task : cores[coreId]) {
            out << " " << task.msgType << " " << task.usrInst;
        }
        out << endl;
    }
    printf("%s", out.str().c_str());
    return 0;
}