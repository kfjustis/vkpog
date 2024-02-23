use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo};
use vulkano::device::physical::{PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo, QueueFlags};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::sync::GpuFuture;
use vulkano::{sync, Validated, Version, VulkanError, VulkanLibrary};
use vulkano::swapchain::{acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use std::sync::Arc;

fn main() {
    // Create an empty event loop.
    let event_loop = EventLoop::new();

    // Get reference to the Vulkan library.
    let library = VulkanLibrary::new().unwrap();

    // Find required extensions for the machine based on the event loop defaults.
    let required_extensions = Surface::required_extensions(&event_loop);

    // Create a Vulkan instance (context) with the portability flag. This allows
    // the app to support devices that don't have a fully compliant Vulkan
    // implementation (e.g. MoltenVK on Apple).
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    // Create a window through winit with a surface to render to.
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    // Create the desired device extensions. This app just needs a swapchain
    // (multi-frame buffer) to render to the window.
    let mut device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    // Find a valid physical device. This app wants at least API version 1.3, but
    // will allow anything with khr_dynamic_rendering. (Any API below 1.3 just has
    // to enable khr_dynamic_rendering manually, as long as it passes the other
    // filters.) This app also prefers at least a graphics-style queue_family
    // and ideally a discrete or integrated GPU.
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            // khr_dynamic_rendering makes it much easier to set up an app that
            // does just a single render pass for each frame.
            p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
        })
        .filter(|p| {
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        }).min_by_key(|(p, _)| {
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5
            }
        })
        .expect("No suitable physical device found.");

    // Debug info so far.
    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    // Remember to enable khr_dynamic_rendering if the API was less than v1.3.
    if physical_device.api_version() < Version::V1_3 {
        device_extensions.khr_dynamic_rendering = true;
    }

    // Create the logical device and get available queues based on the queue
    // family type at the same time.
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            // Just use the one queue that was found to support graphics.
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            enabled_features: Features {
                // v1.3 added this dynamic_rendering feature. Without it, rendering
                // could only be allowed by defining a render pass object. The original
                // triangle.rs example shows how to define a render pass object, but
                // basically this just avoids more boilerplate.
                dynamic_rendering: true,
                ..Features::empty()
            },
            ..Default::default()
        },
    )
    .unwrap();

    // Get the single graphics queue that was already filtered for above.
    let queue = queues.next().unwrap();

    // Now, actually create the swapchain and retrieve the images used by it at
    // the same time.
    let (mut swapchain, images) = {
        // Get the capabilities of the surface to begin with.
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        // Let's view the surface capabilities now.
        println!("\nSurface capabilities: {:?}", surface_capabilities);

        // Set the internal image formats. Can just grab the type from the
        // surface as well.
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        // Create the swapchain and apply the info settings to it.
        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                // Devices almost never use less than 2 images for swaps, but
                // fullscreen always requires at least 2, so use 2 as the min.
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,

                // Note here:
                // Some drivers require the extents (size) of the swapchain to be
                // equal to `surface_capabilities.current_extent`. However, other
                // drivers just set that same variable to `None` and allow whatever
                // extents you want. In the first case, the value is always equal
                // to the inner window size, and in the second case... nothing else
                // besides the window sizer really makes sense...so just use the
                // inner window size no matter what.
                image_extent: window.inner_size().into(),
                // These images will be used as color buffers.
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                // Set whether the window is opaque or transparent.
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };

    // Define a default memory allocator.
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    // Define a vertex buffer with `#[repr(C)]` to guarantee a consistent memory layout.
    // This one will just have a list of vertices for the 3 triangle points.
    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }
    let vertices = [
        Vertex {
            position: [-0.5, 0.5],
        },
        Vertex {
            position: [0.0, -0.5],
        },
        Vertex {
            position: [0.5, 0.5],
        },
    ];
    let vertex_buffer = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices,
    )
    .unwrap();

    // Define the vertex and fragment shaders. Vulkano uses a macro because the
    // Vulkan shader creation process is inherently unsafe on its own.
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450

                layout(location = 0) in vec2 position;

                layout(location = 1) out vec3 vert_color;

                void main() {
                    vert_color = vec3(position.yx, 1.0);

                    gl_Position = vec4(position, 0.0, 1.0);
                }
            ",
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450

                layout(location = 0) out vec4 f_color;

                layout(location = 1) in vec3 vert_color;

                void main() {
                    // f_color = vec4(1.0, 0.0, 0.0, 1.0);
                    f_color = vec4(vert_color, 1.0);
                }
            ",
        }
    }

    // Create a graphics pipeline to draw.
    let pipeline = {
        // Load the shaders.
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        // Generate a vertex input state based on the vertex shader layout.
        let vertex_input_state = Vertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();

        // Manually set the pipeline shader stage order.
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        // Define a pipeline layout. (All shaders can share one layout as long
        // as the layout contains the superset of all needed shader data.)
        let layout = PipelineLayout::new(
            device.clone(),
            // This app only uses one pipeline, so the creation info can be
            // automatically generated from the settings of that pipeline.
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        // Define a subpass that specifies the color attachment formats.
        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(swapchain.image_format())],
            ..Default::default()
        };

        // Now create the actual graphics pipeline with all of the defined settings.
        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                // How vertex data is read from the vertex buffer into the vertex shader.
                vertex_input_state: Some(vertex_input_state),
                // How vertices are arranged nito primitive shapes. (The default
                // primitive shape is a triangle.)
                input_assembly_state: Some(InputAssemblyState::default()),
                // How primitives are transformed and clipped to fit the framebuffer.
                // This app uses a resizable viewport that is set to draw over the
                // entire window.
                viewport_state: Some(ViewportState::default()),
                // How polygons are culled and converted into a raster of pixels.
                // (The default value does not perform any culling.)
                rasterization_state: Some(RasterizationState::default()),
                // How multiple fragment shaderr samples are converted to a single
                // pixel value. (The default value does not perform any multisampling.)
                multisample_state: Some(MultisampleState::default()),
                // How pixel values are combined with the values already present in the
                // framebuffer. (The default value overwrites the old value with the new
                // one, i.e. without any blending.)
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.color_attachment_formats.len() as u32,
                    ColorBlendAttachmentState::default(),
                )),
                // Dynamic states allow specific pipeline settings to be applied when
                // recording the command buffer, before drawing is performed. This setting
                // sets the viewport as dynamic.
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    // Since the viewport is dynamic, only the viewport needs to be recreated when the
    // window is resized. Otherwise, the entire pipeline would need to be recreated.
    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0,
    };

    // Wrap the created images in image views so they can actually be used by the
    // swapchain. Multiple images will be drawn to, so a different image view is
    // needed for each image.
    let mut attachment_image_views = window_size_dependent_setup(&images, &mut viewport);

    // Set up an allocator for the needed command buffers.
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    //////
    // Initialization is finished at this point!
    //////

    // Tracking variable for when the swapchain becomes invalid. Gets reset after the
    // swapchain becomes valid again.
    let mut recreate_swapchain = false;

    // Prepare to submit commands to the GPU. To avoid blocking on the GpuFuture, store
    // the submission of the previous frame.
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    // Now, define the event loop.
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                // Do not draw the frame when the screen size is zero. (This happens
                // on Windows when the app is minimized.)
                let image_extent: [u32; 2] = window.inner_size().into();
                if image_extent.contains(&0) {
                    return;
                }

                // Call cleanup regularly to release unused resources from the GPU.
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // When the window resizes, recreate everything that relies on the window
                // size. In this app, that includes the swapchain, the framebuffers, and
                // the dynamic state viewport.
                if recreate_swapchain {
                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..swapchain.create_info()
                        })
                        .expect("Failed to recreate swapchain.");

                    swapchain = new_swapchain;

                    // Re-wrap the images from the new swapchain in image views so they
                    // can be drawn to.
                    attachment_image_views =
                        window_size_dependent_setup(&new_images, &mut viewport);
                    
                    // Reset the recreate swapchain flag.
                    recreate_swapchain = false;
                }

                // Acquire an image from the swapchain to draw to. (This function blocks
                // if none are available, which can happen if you submit draw commands
                // too quickly.) The function returns the index of the image that can be
                // drawn to.
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {e}"),
                    };

                // If the acquired image happens to be suboptimal, just recreate the swapchain
                // to ensure rendering continues to display correctly. Some drivers return
                // suboptimal when the window resizes, but in that case, it may not mark the
                // swapchain as out of date, so this manual check just ensures stability no
                // matter what.
                if suboptimal {
                    recreate_swapchain = true;
                }

                // Now build the command buffer to draw with. This object holds the commands to
                // execute on the GPU.
                //
                // Note: Building a command buffer is considered `expensive`, but is known to be a
                // an optimized hot path in the driver. Also, a queue family has to be passed to
                // the command buffer and the command buffer will only execute on that family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    // Enter a render pass to the command buffer that matches the render
                    // pass defined in the pipeline earlier.
                    .begin_rendering(RenderingInfo {
                        // Set the color attachment, the image view, and how the image
                        // view should be used.
                        color_attachments: vec![Some(RenderingAttachmentInfo {
                            // Clear the attachment whenever rendering starts.
                            load_op: AttachmentLoadOp::Clear,
                            // Store the rendered output in the attachment image. (This
                            // result could also be discarded if desired.)
                            store_op: AttachmentStoreOp::Store,
                            // The value to clear the attachment with.
                            //
                            // Note: This could be None in an attachment type that doesn't
                            // use AttachmentLoadOp::Clear.
                            clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                            ..RenderingAttachmentInfo::image_view(
                                // Use the image view that matches the currently acquired
                                // swapchain image (denoted by index).
                                attachment_image_views[image_index as usize].clone(),
                            )
                        })],
                        ..Default::default()
                    })
                    .unwrap()

                    // -- This is now inside the first subpass of the render pass.

                    .set_viewport(0, [viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap()
                    // Add the draw command.
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    // And leave the render pass.
                    .end_rendering()
                    .unwrap();

                // Finish building the command buffer by calling build().
                let command_buffer = builder.build().unwrap();

                // Grab the GpuFuture from the render.
                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    // Here, the color output now contains the triangle. Show it with
                    // `then_swapchain_present`.
                    //
                    // Note: This doesn't present the image immediately. Instead it
                    // submits a present command at the end of the queue. This means
                    // that it will only be presented once the GPU has finished
                    // executing the command buffer that draws the triangle.
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                // Assign the frame status based on the future result.
                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {e}");
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    })
}

// Initialize the window images and re-initialize every time the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    viewport: &mut Viewport,
) -> Vec<Arc<ImageView>> {
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];
    images.
        iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>()
}