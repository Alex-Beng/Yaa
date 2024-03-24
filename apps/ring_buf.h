#include <atomic>
#include <vector>

template <typename T>
class LockFreeRingBuffer {
public:
    explicit LockFreeRingBuffer(size_t size)
        : buffer_(size), head_(0), tail_(0) {}

    bool push(const T& value) {
        const auto current_tail = tail_.load(std::memory_order_relaxed);
        const auto next_tail = increment(current_tail);
        if (next_tail != head_.load(std::memory_order_acquire)) {
            buffer_[current_tail] = value;
            tail_.store(next_tail, std::memory_order_release);
            return true;
        }
        return false;  // full queue
    }

    bool pop(T& value) {
        const auto current_head = head_.load(std::memory_order_relaxed);
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false;  // empty queue
        }
        value = buffer_[current_head];
        head_.store(increment(current_head), std::memory_order_release);
        return true;
    }

private:
    size_t increment(size_t idx) const { return (idx + 1) % buffer_.size(); }

    std::vector<T> buffer_;
    std::atomic<size_t> head_, tail_;
};