#include <atomic>
#include <vector>

template <typename T>
class LockFreeRingBuffer {
public:
    explicit LockFreeRingBuffer(size_t size)
        : maxSize(size), buffer(size), head(0), tail(0) {}

    void push(const T& value) {
        auto current_tail = tail.load(std::memory_order_relaxed);
        buffer[current_tail] = value;
        tail.store((current_tail + 1) % maxSize, std::memory_order_release);

        // 如果head即将被覆盖，则将其向前移动
        if ((tail.load(std::memory_order_acquire) + 1) % maxSize == head.load(std::memory_order_relaxed)) {
            // std::cout<<"head is going to be overwritten"<<std::endl;
            head.store((head.load(std::memory_order_relaxed) + 1) % maxSize, std::memory_order_release);
        }
    }

    bool pop(T& value) {
        auto current_head = head.load(std::memory_order_relaxed);
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false; // 缓冲区为空
        }

        value = buffer[current_head];
        head.store((current_head + 1) % maxSize, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return head.load(std::memory_order_acquire) == tail.load(std::memory_order_acquire);
    }

    bool full() const {
        return (tail.load(std::memory_order_acquire) + 1) % maxSize == head.load(std::memory_order_acquire);
    }

    int size() const {
        size_t current_head = head.load(std::memory_order_acquire);
        size_t current_tail = tail.load(std::memory_order_acquire);
        if (current_tail >= current_head) {
            return current_tail - current_head;
        } else {
            return maxSize - current_head + current_tail;
        }
    }

private:
    size_t increment(size_t idx) const { return (idx + 1) % buffer_.size(); }

    std::vector<T> buffer;
    std::atomic<size_t> head;
    std::atomic<size_t> tail;
    const size_t maxSize;
};