// test.cpp

#include <array>
#include <windows.h>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>

uint8_t checksum(const uint8_t *data, size_t len) {
    unsigned sum = 0;
    for (size_t i = 0; i < len; ++i)
        sum += data[i];
    return static_cast<uint8_t>(~sum & 0xFF);
}

std::wstring com_path(const std::string &com) {
    std::wstring w;
    w.assign(com.begin(), com.end());
    return L"\\\\.\\" + w;
}

static HANDLE open_serial_win(const std::string &com) {
    HANDLE h = CreateFileW(com_path(com).c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        std::cerr << "CreateFile failed\n";
        return INVALID_HANDLE_VALUE;
    }

    SetupComm(h, 1 << 16, 1 << 16);
    PurgeComm(h, PURGE_RXABORT | PURGE_RXCLEAR | PURGE_TXABORT | PURGE_TXCLEAR);

    DCB dcb{};
    dcb.DCBlength = sizeof(dcb);
    if (!GetCommState(h, &dcb)) {
        std::cerr << "GetCommState failed\n";
        CloseHandle(h);
        return INVALID_HANDLE_VALUE;
    }

    dcb.BaudRate = 1000000;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fParity = FALSE;
    dcb.fOutxCtsFlow = FALSE;
    dcb.fOutxDsrFlow = FALSE;
    dcb.fDtrControl = DTR_CONTROL_DISABLE;
    dcb.fRtsControl = RTS_CONTROL_DISABLE;
    dcb.fOutX = FALSE;
    dcb.fInX = FALSE;

    if (!SetCommState(h, &dcb)) {
        std::cerr << "SetCommState failed\n";
        CloseHandle(h);
        return INVALID_HANDLE_VALUE;
    }

    COMMTIMEOUTS to{50, 50, 0, 50, 0};
    if (!SetCommTimeouts(h, &to)) {
        std::cerr << "SetCommTimeouts failed\n";
        CloseHandle(h);
        return INVALID_HANDLE_VALUE;
    }

    return h;
}

void put16le(uint8_t *buf, uint16_t v) {
    buf[0] = static_cast<uint8_t>(v);
    buf[1] = static_cast<uint8_t>(v >> 8);
}

static bool read_exact(HANDLE h, uint8_t *buf, DWORD nbytes) {
    DWORD total = 0;
    while (total < nbytes) {
        DWORD got = 0;
        if (!ReadFile(h, buf + total, nbytes - total, &got, nullptr)) {
            std::cerr << "ReadFile failed, err=" << GetLastError() << "\n";
            return false;
        }
        if (got == 0) {
            // Timeout with no data due to COMMTIMEOUTS; let caller decide.
            return false;
        }
        total += got;
    }
    return true;
}

// READ instruction for protocol 1.0
// Request: FF FF ID LEN INST=0x02 ADDR NBYTES CHK
// Response: FF FF ID LEN ERROR [NBYTES DATA] CHK
static bool read_register(HANDLE h, uint8_t id, uint8_t addr, uint8_t nbytes, uint8_t *out) noexcept {
    constexpr uint8_t INST_READ = 0x02;

    uint8_t pkt[8];
    pkt[0] = 0xFF;
    pkt[1] = 0xFF;
    pkt[2] = id;
    pkt[3] = 4; // length = params(2) + 2
    pkt[4] = INST_READ;
    pkt[5] = addr;
    pkt[6] = nbytes;

    uint8_t sum = 0;
    for (size_t i = 2; i <= 6; ++i)
        sum = static_cast<uint8_t>(sum + pkt[i]);
    pkt[7] = static_cast<uint8_t>(~sum);

    DWORD written = 0;
    if (!WriteFile(h, pkt, sizeof(pkt), &written, nullptr) || written != sizeof(pkt)) {
        std::cerr << "WriteFile(read) failed, err=" << GetLastError() << "\n";
        return false;
    }
    FlushFileBuffers(h);

    // Read status packet header (2 + 1 + 1 = 4) then tail
    uint8_t hdr[4];
    if (!read_exact(h, hdr, 4)) {
        std::cerr << "Timeout waiting for status header\n";
        return false;
    }
    if (!(hdr[0] == 0xFF && hdr[1] == 0xFF)) {
        std::cerr << "Bad header\n";
        return false;
    }
    if (hdr[2] != id) {
        std::cerr << "Unexpected ID in response\n";
        return false;
    }
    uint8_t len = hdr[3]; // = error(1) + params(N) + 2?
    // For protocol 1.0: total remaining bytes to read = len + 1 (checksum)  (because 'len' includes error+params+1 for checksum? No.)
    // Standard: LENGTH = PARAMS + 2 (Error + Checksum). So after the 4 bytes, we still need to read 'len' bytes.
    // We'll read 'len' bytes (error + params + checksum).
    std::vector<uint8_t> body(len);
    if (!read_exact(h, body.data(), len)) {
        std::cerr << "Timeout waiting for status body\n";
        return false;
    }

    if (body.empty()) {
        std::cerr << "Empty body\n";
        return false;
    }

    uint8_t error = body[0];
    if (error != 0) {
        std::cerr << "Servo error: 0x" << std::hex << int(error) << std::dec << "\n";
        return false;
    }

    // body layout: [error][params...][checksum]; params length = len - 2
    if (len < 2) {
        std::cerr << "Malformed length\n";
        return false;
    }
    uint8_t params_len = static_cast<uint8_t>(len - 2);
    if (params_len < nbytes) {
        std::cerr << "Not enough data\n";
        return false;
    }

    // Verify checksum over ID + LENGTH + ERROR + PARAMS
    uint8_t csum_calc = 0;
    csum_calc += hdr[2];
    csum_calc += hdr[3];
    for (uint8_t i = 0; i < len - 1; ++i)
        csum_calc = static_cast<uint8_t>(csum_calc + body[i]);
    csum_calc = static_cast<uint8_t>(~csum_calc);
    uint8_t csum_rx = body[len - 1];
    if (csum_calc != csum_rx) {
        std::cerr << "Checksum mismatch\n";
        return false;
    }

    // Copy first nbytes from params
    for (uint8_t i = 0; i < nbytes; ++i)
        out[i] = body[1 + i];
    return true;
}

static bool read_present_position(HANDLE h, uint8_t id, uint16_t &pos) noexcept {
    constexpr uint8_t ADDR_PRESENT_POSITION_L = 0x38;
    uint8_t buf[2] = {0, 0};
    if (!read_register(h, id, ADDR_PRESENT_POSITION_L, 2, buf))
        return false;
    pos = static_cast<uint16_t>(buf[0] | (static_cast<uint16_t>(buf[1]) << 8));
    return true;
}

// https://files.waveshare.com/upload/2/27/Communication_Protocol_User_Manual-EN%28191218-0923%29.pdf
static bool write_pos(HANDLE h, uint8_t id, uint16_t pos, uint16_t time_ms, uint16_t speed) noexcept {
    constexpr uint8_t INST_WRITE = 0x03;
    constexpr uint8_t ADDR_GOAL_POSITION_L = 0x2A;

    std::array<uint8_t, 13> pkt{};
    pkt[0] = 0xFF;
    pkt[1] = 0xFF;
    pkt[2] = id;
    pkt[3] = 9; // 3 + 6
    pkt[4] = INST_WRITE;
    pkt[5] = ADDR_GOAL_POSITION_L;
    put16le(&pkt[6], pos);
    put16le(&pkt[8], time_ms);
    put16le(&pkt[10], speed);

    // Protocol 1.0 checksum = ~(id + length + instruction + params) & 0xFF
    uint8_t sum = 0;
    for (size_t i = 2; i <= 11; ++i)
        sum = static_cast<uint8_t>(sum + pkt[i]);
    pkt[12] = static_cast<uint8_t>(~sum);

    DWORD written = 0;
    if (!WriteFile(h, pkt.data(), pkt.size(), &written, nullptr) ||
        written != pkt.size()) {
        std::cerr << "WriteFile failed, err=" << GetLastError() << "\n";
        return false;
    }
    FlushFileBuffers(h);
    return true;
}

static bool wait_until_reached(HANDLE h, uint8_t id, uint16_t goal, uint16_t tolerance, unsigned timeout_ms,
                               unsigned poll_ms = 30) {
    auto start = std::chrono::steady_clock::now();
    while (true) {
        uint16_t present = 0;
        if (read_present_position(h, id, present)) {
            int err = static_cast<int>(present) - static_cast<int>(goal);
            if (err < 0)
                err = -err;
            // Uncomment to see convergence
            // std::cout << "present=" << present << " goal=" << goal << " err=" << err << "\n";
            if (err <= static_cast<int>(tolerance))
                return true;
        } else {
            // optional: keep trying until timeout
        }

        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start)
            .count() > timeout_ms) {
            std::cerr << "Timeout waiting for position.\n";
            return false;
        }
        Sleep(poll_ms);
    }
}

int main(int argc, char **argv) {
    std::string com = "COM3";
    uint8_t motor_id = 6;
    uint16_t pos_A = 1960;
    uint16_t pos_B = 3389;
    uint16_t time_ms = 0;
    uint16_t speed = 0;

    // https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    // https://foxglove.dev/blog/visualizing-lerobot-so-100-using-foxglove?utm_source=chatgpt.com
    // https://github.com/huggingface/lerobot/blob/main/src/lerobot/model/kinematics.py

    HANDLE h = open_serial_win(com);
    if (h == INVALID_HANDLE_VALUE) {
        std::cerr << "Error opening serial port " << com << "\n";
        return 1;
    }
    std::cout << "Opened " << com << "\n";

    const int cycles = 5;
    const uint16_t tolerance = 10;
    const unsigned settle_timeout_ms = 3000;

    uint16_t goals[2] = {pos_A, pos_B};
    int gi = 0;

    for (int c = 0; c < cycles; ++c) {
        uint16_t goal = goals[gi];
        std::cout << "[Cycle " << (c + 1) << "] Move to " << goal << "\n";

        if (!write_pos(h, motor_id, goal, time_ms, speed)) {
            std::cerr << "Failed to send WRITE position.\n";
            CloseHandle(h);
            return 1;
        }

        if (!wait_until_reached(h, motor_id, goal, tolerance, settle_timeout_ms)) {
            std::cerr << "Did not converge to goal within timeout.\n";
            CloseHandle(h);
            return 1;
        }
        std::cout << "Reached " << goal << "\n";
        Sleep(200);

        gi ^= 1;
    }

    CloseHandle(h);
    std::cout << "Done.\n";
    return 0;
}
