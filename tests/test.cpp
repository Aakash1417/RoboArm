#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <vector>

// STS3215 Protocol constants
#define HEADER 0xFF
#define BROADCAST_ID 0xFE

// Instruction set
#define WRITE_INSTRUCTION 0x03

// Control table addresses
#define GOAL_POSITION_L 0x2A // Goal position low byte
#define GOAL_POSITION_H 0x2B // Goal position high byte

int openSerial(const char *portName, int baudrate = B1000000)
{
    int fd = open(portName, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1)
    {
        perror("Unable to open port");
        return -1;
    }

    termios options;
    tcgetattr(fd, &options);
    cfsetispeed(&options, baudrate);
    cfsetospeed(&options, baudrate);
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    tcsetattr(fd, TCSANOW, &options);

    return fd;
}

uint8_t calcChecksum(const std::vector<uint8_t> &packet)
{
    int sum = 0;
    for (size_t i = 2; i < packet.size(); i++)
    {
        sum += packet[i];
    }
    return (~sum) & 0xFF;
}

// Send move command
void moveServo(int fd, uint8_t id, uint16_t position)
{
    std::vector<uint8_t> packet;

    packet.push_back(HEADER);
    packet.push_back(HEADER);
    packet.push_back(id); // Servo ID
    packet.push_back(5);  // Length = 5 (Instruction + address + 2 data)
    packet.push_back(WRITE_INSTRUCTION);
    packet.push_back(GOAL_POSITION_L);

    packet.push_back(position & 0xFF);        // Low byte
    packet.push_back((position >> 8) & 0xFF); // High byte

    packet.push_back(calcChecksum(packet));

    int result = write(fd, packet.data(), packet.size());
    if (result < 0)
        perror("Failed to write to serial port");

    usleep(1000);
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <port> <servo_id>" << std::endl;
        return -1;
    }
    const char *portName = argv[1];
    uint8_t servoId = std::stoi(argv[2]);

    int fd = openSerial(portName, B1000000);
    if (fd < 0)
        return -1;

    std::cout << "port opened" << std::endl;

    std::cout << "File descriptor: " << fd << std::endl;

    moveServo(fd, servoId, 0);

    std::cout << "Commands sent!" << std::endl;

    close(fd);
    return 0;
}
