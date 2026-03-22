package main

import "fmt"

func main() {
	calc := NewCalculator()
	result := calc.Add(3, 5)
	fmt.Println("Result:", result)

	greeting := Greet("World")
	fmt.Println(greeting)
}
