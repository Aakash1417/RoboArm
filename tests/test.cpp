// test.cpp

#include <array>
#include <windows.h>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>

uint8_t checksum(const uint8_t *data, size_t len) {
    unsigned sum = 0;
    for (size_t i = 0; i < len; ++i) sum += data[i];
    return static_cast<uint8_t>(~sum & 0xFF);
}

std::wstring com_path(const std::string &com) {
    // Win32 requires the \\.\ prefix for COM10+ (works for all)
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

// https://manuals.plus/sv/feetech/scs15-bus-smart-control-servo-manual
// https://files.waveshare.com/upload/2/27/Communication_Protocol_User_Manual-EN%28191218-0923%29.pdf
static bool write_pos(HANDLE h, uint8_t id, uint16_t pos, uint16_t time_ms, uint16_t speed) noexcept {
    constexpr uint8_t INST_WRITE = 0x03;
    constexpr uint8_t ADDR_GOAL_POSITION_L = 0x2A;

    std::array<uint8_t, 13> pkt{};
    pkt[0] = 0xFF;
    pkt[1] = 0xFF;
    pkt[2] = id;
    pkt[3] = 8; // 2 + 6
    pkt[4] = INST_WRITE;
    pkt[5] = ADDR_GOAL_POSITION_L;
    pkt[6] = static_cast<uint8_t>(pos);
    pkt[7] = static_cast<uint8_t>(pos >> 8);
    pkt[8] = static_cast<uint8_t>(time_ms);
    pkt[9] = static_cast<uint8_t>(time_ms >> 8);
    pkt[10] = static_cast<uint8_t>(speed);
    pkt[11] = static_cast<uint8_t>(speed >> 8);

    // Protocol 1.0 checksum = ~(id + length + instruction + params) & 0xFF
    uint8_t sum = 0;
    for (size_t i = 2; i <= 11; ++i) sum = static_cast<uint8_t>(sum + pkt[i]);
    pkt[12] = static_cast<uint8_t>(~sum);

    DWORD written = 0;
    if (!WriteFile(h, pkt.data(), static_cast<DWORD>(pkt.size()), &written, nullptr) ||
        written != pkt.size()) {
        std::cerr << "WriteFile failed, err=" << GetLastError() << "\n";
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    std::string com = "COM3";

    uint8_t motor_id = 6;
    uint16_t position = 1960;
    position = 3389;
    uint16_t time_ms = 0;
    uint16_t speed = 0;

    HANDLE h = open_serial_win(com);
    if (h == INVALID_HANDLE_VALUE) {
        std::cerr << "Error opening serial port" << com << "\n";
    }
    std::cout << "Opened " << com << "\n";

    bool ok = write_pos(h, motor_id, position, time_ms, speed);
    CloseHandle(h);

    if (!ok) {
        std::cerr << "Failed to send command.\n";
        return 1;
    }
    std::cout << "Command sent.\n";
    return 0;
}
