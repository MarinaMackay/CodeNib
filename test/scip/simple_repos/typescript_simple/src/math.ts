/**
 * Math utility functions
 */

export function add(a: number, b: number): number {
    return a + b;
}

export function multiply(a: number, b: number): number {
    return a * b;
}

export function subtract(a: number, b: number): number {
    return a - b;
}

export class Calculator {
    constructor(public name: string) {}

    calculate(x: number, y: number): number {
        return add(x, y);
    }
}
