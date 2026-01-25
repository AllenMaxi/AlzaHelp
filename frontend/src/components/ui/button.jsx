import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-3 whitespace-nowrap rounded-xl text-lg font-semibold ring-offset-background transition-all duration-300 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-6 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90 shadow-soft hover:shadow-card active:scale-[0.98]",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/90 shadow-soft hover:shadow-card active:scale-[0.98]",
        outline:
          "border-2 border-primary bg-background text-primary hover:bg-primary hover:text-primary-foreground shadow-soft hover:shadow-card active:scale-[0.98]",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80 shadow-soft hover:shadow-card active:scale-[0.98]",
        ghost: "hover:bg-accent hover:text-accent-foreground active:scale-[0.98]",
        link: "text-primary underline-offset-4 hover:underline",
        // Accessible variants for Alzheimer's app - extra large and visible
        accessible: "bg-primary text-primary-foreground hover:bg-primary/90 shadow-card hover:shadow-elevated min-h-[4.5rem] min-w-[14rem] text-xl font-bold rounded-2xl active:scale-[0.97]",
        "accessible-secondary": "bg-secondary text-secondary-foreground hover:bg-secondary/80 shadow-card hover:shadow-elevated min-h-[4.5rem] min-w-[14rem] text-xl font-bold rounded-2xl active:scale-[0.97]",
        "accessible-outline": "border-3 border-primary bg-background text-primary hover:bg-primary/10 shadow-card hover:shadow-elevated min-h-[4.5rem] min-w-[14rem] text-xl font-bold rounded-2xl active:scale-[0.97]",
        // Family member button variants
        family: "bg-family-spouse/20 text-foreground border-2 border-family-spouse hover:bg-family-spouse/30 shadow-soft hover:shadow-card min-h-[5rem] rounded-2xl active:scale-[0.98]",
        success: "bg-success text-success-foreground hover:bg-success/90 shadow-soft hover:shadow-card active:scale-[0.98]",
      },
      size: {
        default: "h-14 px-6 py-3",
        sm: "h-11 rounded-lg px-4 text-base",
        lg: "h-16 rounded-xl px-8 text-xl",
        xl: "h-20 rounded-2xl px-10 text-2xl",
        icon: "h-14 w-14 rounded-xl",
        "icon-lg": "h-18 w-18 rounded-2xl",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

const Button = React.forwardRef(({ className, variant, size, asChild = false, ...props }, ref) => {
  const Comp = asChild ? Slot : "button"
  return (
    <Comp
      className={cn(buttonVariants({ variant, size, className }))}
      ref={ref}
      {...props}
    />
  )
})
Button.displayName = "Button"

export { Button, buttonVariants }
