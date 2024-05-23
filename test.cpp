#include <tbb/flow_graph.h>
#include <iostream>

int main()
{
    // 创建一个图对象
    tbb::flow::graph g;

    // 创建一个可以发出数据的广播节点
    tbb::flow::broadcast_node<int> inputNode(g);

    // 创建一个可以接收数据并对其执行操作的函数节点
    tbb::flow::function_node<int, int> printNode(g, 1, [](const int& n) {
        std::cout << "Received " << n << std::endl;
        return n;  // 返回接收到的数
    });

    // 创建一个可以接收数据并对其执行操作的函数节点，计算平方
    tbb::flow::function_node<int, int> squareNode(g, 1, [](const int& n) {
        std::cout << "Square of " << n << " is " << n * n << std::endl;
        return n * n;  // 返回数的平方
    });

    // 将广播节点和函数节点连接起来，这意味着当广播节点发出数据时，函数节点会接收并处理它
    tbb::flow::make_edge(inputNode, printNode);
    tbb::flow::make_edge(printNode, squareNode);

    // 让广播节点发送一些数据
    inputNode.try_put(10);
    inputNode.try_put(20);
    inputNode.try_put(30);

    // 等待所有任务完成
    g.wait_for_all();

    return 0;
}