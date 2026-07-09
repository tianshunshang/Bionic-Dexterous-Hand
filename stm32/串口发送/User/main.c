#include "stm32f10x.h"
#include "Delay.h"
#include "OLED.h"
#include "Serial.h"
#include "Servo.h"

/*
 * 协议：Python 端发送 1 字节，值即为舵机目标角度（0~180）
 *   中指完全伸直 → 0°
 *   中指完全弯曲 → 180°
 */

int main(void)
{
    uint8_t RxData    = 0;
    uint8_t ServoAngle = 0;

    /*---- 模块初始化 ----*/
    OLED_Init();
    Serial_Init();
    Servo_Init();

    /*---- OLED 静态标签 ----*/
    OLED_ShowString(1, 1, "Middle:");
    OLED_ShowString(2, 1, "Servo: ");

    /*---- 舵机复位到 0° ----*/
    Servo_SetAngle(0.0f);
    Delay_ms(500);

    while (1)
    {
        if (Serial_GetRxFlag() == 1)
        {
            RxData = Serial_GetRxData();

            /* 范围校验：仅接受 0~180 */
            if (RxData <= 180)
            {
                ServoAngle = RxData;

                /* 驱动舵机 */
                Servo_SetAngle((float)ServoAngle);

                /* OLED 更新 */
                OLED_ShowNum(1, 8, ServoAngle, 3);   // 中指弯曲角度（舵机角）
                OLED_ShowNum(2, 8, ServoAngle, 3);   // 舵机当前角度
            }
        }
    }
}
