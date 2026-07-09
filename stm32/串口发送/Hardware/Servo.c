#include "stm32f10x.h"
#include "Servo.h"

/*
 * 舵机驱动模块
 * 使用 TIM2_CH2 输出 PWM，对应引脚 PA1
 *
 * 舵机 PWM 参数：
 *   周期   = 20ms  (50Hz)
 *   脉宽   = 0.5ms (对应  0°) ~ 2.5ms (对应 180°)
 *
 * TIM2 配置：
 *   APB1 时钟 = 72MHz（经 x2 后为 72MHz，此处直接用72MHz）
 *   预分频器  PSC = 72 - 1   → 计数频率 = 1MHz (1us/tick)
 *   自动重载  ARR = 20000 - 1 → 周期     = 20ms
 *   比较值    CCR2 范围：500（0°）~ 2500（180°）
 */

/**
  * 函    数：舵机初始化
  * 参    数：无
  * 返 回 值：无
  */
void Servo_Init(void)
{
    /*---- 开启时钟 ----*/
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);   // GPIOA
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2,  ENABLE);   // TIM2

    /*---- PA1 复用推挽输出（TIM2_CH2）----*/
    GPIO_InitTypeDef GPIO_InitStructure;
    GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_AF_PP;
    GPIO_InitStructure.GPIO_Pin   = GPIO_Pin_1;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    /*---- TIM2 时基配置 ----*/
    TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
    TIM_TimeBaseStructure.TIM_ClockDivision     = TIM_CKD_DIV1;
    TIM_TimeBaseStructure.TIM_CounterMode       = TIM_CounterMode_Up;
    TIM_TimeBaseStructure.TIM_Period            = 20000 - 1;   // ARR: 20ms 周期
    TIM_TimeBaseStructure.TIM_Prescaler         = 72 - 1;      // PSC: 1MHz 计数
    TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
    TIM_TimeBaseInit(TIM2, &TIM_TimeBaseStructure);

    /*---- TIM2 PWM 输出比较配置（通道2）----*/
    TIM_OCInitTypeDef TIM_OCInitStructure;
    TIM_OCStructInit(&TIM_OCInitStructure);                    // 清零默认值
    TIM_OCInitStructure.TIM_OCMode      = TIM_OCMode_PWM1;
    TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
    TIM_OCInitStructure.TIM_Pulse       = 1500;                // 初始90°
    TIM_OCInitStructure.TIM_OCPolarity  = TIM_OCPolarity_High;
    TIM_OC2Init(TIM2, &TIM_OCInitStructure);

    /*---- 使能 CCR2 预装载，并启动定时器 ----*/
    TIM_OC2PreloadConfig(TIM2, TIM_OCPreload_Enable);
    TIM_ARRPreloadConfig(TIM2, ENABLE);
    TIM_Cmd(TIM2, ENABLE);
}

/**
  * 函    数：设置舵机角度
  * 参    数：Angle 目标角度，范围 0.0 ~ 180.0 度
  * 返 回 值：无
  */
void Servo_SetAngle(float Angle)
{
    /* 将角度线性映射到 PWM 脉宽
     * 0°   -> CCR = 500  (0.5ms)
     * 180° -> CCR = 2500 (2.5ms)
     */
    uint16_t CCR_Value = (uint16_t)(Angle / 180.0f * 2000.0f + 500.0f);
    TIM_SetCompare2(TIM2, CCR_Value);
}
